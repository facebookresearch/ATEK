# ATEK Evaluation libraries

To evaluate ML model performances, a common workflow is:

- Run model on some test datasets
- Save the groundtruth and model prediction results to local files
- Write some evaluation metric functions.
- Write a benchmarking script that parses in the local results, and use the hand-crafted metric functions to compute the metrics.

In ATEK, we include such standarized evaluation workflows for various [Aria-related machine perception tasks](./ml_tasks.md). For each of the supported ML task, ATEK provides the following:

- **Evaluation datasets** that user can direct download.
- **A standardized prediction file format**, user just need to write a simple writer class to parse both groundtruth and their model output into this format .
- A library consisting of **common metrics functions** for specific tasks.
- An easy-to-run **benchmarking script**, which takes the groundtruth and prediction files, and reports common metrics for the task.

Currently we supports 2 perception ML tasks for Aria:
- [static 3D object detection](./ML_task_object_detection.md)
- [3D surface reconstruction](./ML_task_surface_recon.md)
