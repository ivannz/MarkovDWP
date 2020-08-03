# ToDo

The list of fixed to do while the experiments are running:
* sometimes the logs made by the `reconstruction_logger.py` are attributed to incorrect steps. This depends mostly on the `batch_size` setting. Solution: make them use eother `current_epoch` timing or a completely independent one (commit=False was used deliberately to send the logs on the next call to `trainer.log_metrics`). Think about making rec-logger wandb-inpdependent.

TRIP: forward pass through the sum-reduced cores is not-normalized and suffers from floating point underflow.
* ~`trip_index_sample`~, ~`TRIP.log_prob`~ and ~`trip_index_log_prob`~ head is not unit-normalized
* ~`trip_index_log_marginals` requires careful tracking of max-normalization for the correct renorm of the marginals~
