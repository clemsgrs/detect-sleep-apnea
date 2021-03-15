import numpy as np

from .metrics import precision_function, recall_function, f1_function
from .metric_dreem import custom_metric_function

# available_score_functions = {
#     "precision": precision_function(),
#     "recall": recall_function(),
#     "f1": f1_function(),
# }

available_score_functions = {
    "custom_metric": custom_metric_function(),
}

def compute_metrics_dataset(
        network,
        test_dataset,
        threshold,
        # test_metrics=["precision", "recall", "f1"],
        test_metrics=["custom_metric"],
):
    """
    Computes metrics on current net for test_dataset, using threshold
    as classification threshold
    """

    metrics = {
        score: score_function for score, score_function in
        available_score_functions.items() if score in test_metrics
    }

    metrics_test = [{
        metric: []
        for metric in metrics.keys()
    } for _ in range(network.number_of_classes - 1)]

    all_predicted_events = network.predict_dataset(
        test_dataset,
        threshold,
        batch_size=128)

    found_some_events = False

    for event_num in range(network.number_of_classes - 1):

        for record in test_dataset.records:

            # Select current event predictions
            predicted_events = all_predicted_events[record][event_num]

            # If no predictions skip record, else some_events = 1
            if len(predicted_events) == 0:
                continue

            found_some_events = True

            # Select current true events
            events = test_dataset.get_record_events(record)[event_num]
            # print(f'\ntype(predicted_events): {type(predicted_events)}')
            # print(f'type(events): {type(events)}')
            # print(f'predicted_events: {predicted_events}')
            # print(f'events: {events}')
            events = from_start_duration_to_y(events)
            predicted_events = from_start_duration_to_y(predicted_events)
            
            # Compute_metrics(events, predicted_events, threshold)
            for metric in metrics.keys():
                metrics_test[event_num][metric].append(metrics[metric](
                    predicted_events,
                    events))

    # If for any event and record the network predicted events, return -1
    if found_some_events is False:
        return -1

    for metric in metrics.keys():
        metrics_test[event_num][metric] = np.nanmean(
            np.array(metrics_test[event_num][metric]))

    return metrics_test


def from_start_duration_to_y(events):
    y = np.zeros(90)
    if len(events) == 0:
        return y
    else:
        for i, (s,d) in enumerate(events):
            y[s//100:s//100+d//100] = 1
        return y
