import json
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from common.hierarchical_logger import hlog, htrack
from common.request import Request, RequestResult
from .scenario import Instance, Scenario, Reference, TRAIN_TAG, VALID_TAG, TEST_TAG


@dataclass(frozen=True)
class AdapterSpec:
    """
    Specifies how to take an `Instance` and produce a set of `Request`s (e.g.,
    concatenate instructions and number of training examples) and make one
    request for each reference output).
    """

    instructions: str  # Prompt starts with instructions.
    max_train_instances: int  # Maximum number of (in-context) training instances to put into the prompt
    max_eval_instances: int  # Maximum number of evaluation instances (only reduce this for piloting)
    num_outputs: int  # Generate this many outputs per request

    num_train_trials: int  # Number of random training instances we want to randomize over

    # Decoding parameters
    model: str  # Name of the model we want to query
    temperature: float  # Temperature to use
    stop_sequences: List[str]  # When to stop

    def __str__(self) -> str:
        return (
            f"Instructions: {self.instructions}\n"
            f"Maximum train instances: {self.max_train_instances}\n"
            f"Maximum eval instances: {self.max_eval_instances}\n"
            f"Number of outputs: {self.num_outputs}\n"
            f"Number of train trials: {self.num_train_trials}\n\n"
            f"Model: {self.model}\n"
            f"Temperature: {self.temperature}\n"
            f"Stop sequences: {', '.join(self.stop_sequences)}\n"
        )

    def to_dict(self) -> Dict:
        return {
            "instructions": self.instructions,
            "max_train_instances": self.max_train_instances,
            "max_eval_instances": self.max_eval_instances,
            "num_outputs": self.num_outputs,
            "num_train_trials": self.num_train_trials,
            "model": self.model,
            "temperature": self.temperature,
            "stop_sequences": self.stop_sequences,
        }


@dataclass(frozen=True)
class RequestState:
    """
    A `RequestState` represents a single `Request` made on behalf of an `Instance`.
    It should have all the information that's needed later for a `Metric` to be
    able to understand the `Request` and its `RequestResult`.
    """

    instance: Instance  # Which instance we're evaluating
    reference_index: Optional[int]  # Which reference of the instance we're evaluating (if any)
    train_trial_index: int  # Which training set
    request: Request  # The request that is synthesized
    result: Optional[RequestResult]  # Filled in when we make the call

    def __str__(self) -> str:
        output = f"Train trial index: {self.train_trial_index}\n"
        if self.reference_index:
            output += f"Reference index: {self.reference_index}\n"

        output += f"Instance\n{self.instance}\n\n"
        output += f"Request\n{self.request}\n\n"
        if self.result:
            output += f"Request result\n{self.result}"
        return output

    def to_dict(self) -> Dict:
        result = {
            "instance": self.instance.to_dict(),
            "train_trial_index": self.train_trial_index,
            "request": self.request.to_dict(),
        }
        if self.reference_index:
            result["reference_index"] = self.reference_index
        if self.result:
            result["result"] = self.result.to_dict()
        return result


@dataclass
class ScenarioState:
    """All the `RequestState` results that come about from evaluating a particular scenario."""

    adapter_spec: AdapterSpec
    request_states: List[RequestState]

    def __str__(self) -> str:
        total: int = len(self.request_states)
        output: str = f"Adapter\n{self.adapter_spec}\n{total} request states"
        for i, request_state in enumerate(self.request_states):
            output += f"\n\n------- Request state {i + 1}/{total}\n{request_state}"
        return output

    def __post_init__(self):
        # Create an index for `instances` and `request_states`.
        self.instances = []
        self.request_state_map: Dict[Tuple[int, Instance, Optional[int]], List[RequestState]] = defaultdict(list)
        instances_set = set()
        for request_state in self.request_states:
            instances_set.add(request_state.instance)
            key = (request_state.train_trial_index, request_state.instance, request_state.reference_index)
            self.request_state_map[key].append(request_state)
        self.instances = list(instances_set)

    def get_request_states(
        self, train_trial_index: int, instance: Instance, reference_index: Optional[int]
    ) -> List[RequestState]:
        return self.request_state_map.get((train_trial_index, instance, reference_index), [])

    def to_dict(self) -> Dict:
        return {
            "adapter_spec": self.adapter_spec.to_dict(),
            "request_states": [request_state.to_dict() for request_state in self.request_states],
        }

    def to_json(self, pretty=False) -> str:
        """
        Converts `ScenarioState` into JSON string.
        """
        return json.dumps(self.to_dict(), indent=4) if pretty else json.dumps(self.to_dict())


class Adapter:
    """An `Adapter`"""

    def __init__(self, adapter_spec: AdapterSpec):
        self.adapter_spec = adapter_spec

    @htrack(None)
    def adapt(self, scenario: Scenario) -> ScenarioState:
        """
        Takes a `Scenario` containing a list of instances and builds a list of
        corresponding request_states.  The reason we don't do this per (eval)
        instance is that we create a common set of training instances which is
        shared across all eval instances.
        """
        # Create instances
        instances = scenario.get_instances()

        # Choose training instances and evaluation instances
        all_train_instances = [instance for instance in instances if TRAIN_TAG in instance.tags]
        if len(all_train_instances) < self.adapter_spec.max_train_instances:
            hlog(
                f"WARNING: only {len(all_train_instances)} training instances, "
                f"wanted {self.adapter_spec.max_train_instances}"
            )
        eval_instances = [instance for instance in instances if VALID_TAG in instance.tags or TEST_TAG in instance.tags]
        hlog(
            f"{len(instances)} instances, "
            f"choosing {self.adapter_spec.max_train_instances}/{len(all_train_instances)} train instances, "
            f"{len(eval_instances)} eval instances"
        )

        request_states: List[RequestState] = []

        for train_trial_index in range(self.adapter_spec.num_train_trials):
            # Choose a random set of training instances
            random.seed(train_trial_index)
            train_instances = random.sample(
                all_train_instances, min(len(all_train_instances), self.adapter_spec.max_train_instances)
            )

            # Create request_states
            for eval_instance in eval_instances:

                def process(reference_index: Optional[int], reference: Optional[Reference]):
                    prompt = self.construct_prompt(train_instances, eval_instance, reference)
                    request = Request(
                        model=self.adapter_spec.model,
                        prompt=prompt,
                        temperature=self.adapter_spec.temperature,
                        # TODO: if using single token, then set top_k_per_token instead
                        num_completions=self.adapter_spec.num_outputs,
                        stop_sequences=self.adapter_spec.stop_sequences,
                    )
                    request_state = RequestState(
                        instance=eval_instance,
                        reference_index=reference_index,
                        train_trial_index=train_trial_index,
                        request=request,
                        result=None,
                    )
                    request_states.append(request_state)

                # Request without reference (free-form generation)
                process(None, None)

                # Request for each reference
                for reference_index, reference in enumerate(eval_instance.references):
                    process(reference_index, reference)

        hlog(f"{len(request_states)} requests")
        return ScenarioState(self.adapter_spec, request_states)

    def construct_prompt(
        self, train_instances: List[Instance], eval_instance: Instance, reference: Optional[Reference]
    ) -> str:
        """
        Returns a prompt (string) given `self.adapter_spec.instructions`,
        `train_instances` (in-context training examples), the input part of the
        `eval_instance`, and optionally the `reference`.
        """
        # TODO: support input + output formats
        # TODO: make this configurable if desired
        # TODO: what happens if we have multiline text?
        input_prefix = "Input: "
        output_prefix = "Output: "

        # Instructions
        lines = [self.adapter_spec.instructions]

        # In-context training instances
        for instance in train_instances:
            lines.append("")
            lines.append(input_prefix + instance.input)
            # Put only the correct reference as the output
            reference = instance.first_correct_reference
            if reference is not None:
                lines.append(output_prefix + reference.output)
            else:
                hlog(f"WARNING: no correct reference for {instance}")
                lines.append(output_prefix + "???")

        # Evaluation instance
        lines.append(input_prefix + eval_instance.input)
        # TODO: removing trailing whitespace
        if reference is None:
            lines.append(output_prefix)
        else:
            lines.append(output_prefix + reference.output)

        return "\n".join(lines)
