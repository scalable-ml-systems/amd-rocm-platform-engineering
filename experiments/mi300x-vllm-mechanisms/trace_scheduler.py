from __future__ import annotations

import json

from vllm.v1.core.sched.scheduler import Scheduler


class TraceScheduler(Scheduler):
    """
    Observe vLLM's real scheduling decisions without changing them.

    Every call to schedule() represents one scheduler iteration.

    We call the normal vLLM scheduler first, then print a compact
    JSON record describing what vLLM decided to execute.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._trace_step = 0


    def schedule(self, *args, **kwargs):
        scheduler_output = super().schedule(*args, **kwargs)

        self._trace_step += 1

        scheduled_tokens_by_request = dict(
            scheduler_output.num_scheduled_tokens
        )

        new_request_ids = [
            request.req_id
            for request in scheduler_output.scheduled_new_reqs
        ]

        cached_request_ids = list(
            scheduler_output.scheduled_cached_reqs.req_ids
        )

        resumed_request_ids = sorted(
            scheduler_output.scheduled_cached_reqs.resumed_req_ids
        )

        finished_request_ids = sorted(
            scheduler_output.finished_req_ids
        )

        request_states = self._build_request_states(
            scheduler_output=scheduler_output,
        )

        trace_record = {
            "trace_type": "scheduler_iteration",
            "step": self._trace_step,

            "total_scheduled_tokens":
                scheduler_output.total_num_scheduled_tokens,

            "scheduled_request_count":
                len(scheduled_tokens_by_request),

            "scheduled_tokens_by_request":
                scheduled_tokens_by_request,

            "new_request_ids":
                new_request_ids,

            "cached_request_ids":
                cached_request_ids,

            "resumed_request_ids":
                resumed_request_ids,

            "finished_request_ids":
                finished_request_ids,

            "request_states":
                request_states,
        }

        print(
            "[BATCH_TRACE] "
            + json.dumps(
                trace_record,
                sort_keys=True,
            ),
            flush=True,
        )

        return scheduler_output


    def _build_request_states(
        self,
        scheduler_output,
    ) -> dict[str, dict]:
        """
        Build readable state for requests already known to the worker.

        CachedRequestData gives us:
        - request ID
        - tokens computed before this scheduler step
        - output tokens already generated

        Combined with num_scheduled_tokens, this helps us distinguish
        large prefill/chunk work from normal one-token decode work.
        """

        cached_requests = (
            scheduler_output.scheduled_cached_reqs
        )

        request_states = {}

        for (
            request_id,
            num_computed_tokens,
            num_output_tokens,
        ) in zip(
            cached_requests.req_ids,
            cached_requests.num_computed_tokens,
            cached_requests.num_output_tokens,
        ):

            num_tokens_scheduled_now = (
                scheduler_output.num_scheduled_tokens.get(
                    request_id,
                    0,
                )
            )

            request_states[request_id] = {
                "num_computed_tokens_before_step":
                    num_computed_tokens,

                "num_output_tokens_before_step":
                    num_output_tokens,

                "num_tokens_scheduled_now":
                    num_tokens_scheduled_now,
            }

        return request_states
