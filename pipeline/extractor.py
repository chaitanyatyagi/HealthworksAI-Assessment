def aggregate_results(results):
    """
    Flattens the list of lists returned by the batch pipeline.
    Since extract_from_chunk already returns parsed objects,
    no further json.loads() is needed.
    """
    final_data = []

    for res in results:
        # res is the list of dicts returned from one chunk
        if isinstance(res, list):
            final_data.extend(res)
        elif isinstance(res, dict):
            final_data.append(res)

    return final_data