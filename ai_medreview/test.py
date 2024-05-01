from tqdm import tqdm
import time  # Used to simulate delays in processing for demonstration

if __name__ == "__main__":
    # Define tasks as a list of functions to be executed
    tasks = [
        load_google_sheet,
        load_local_data,
        process_new_data,
        anonymize_data,
        preprocess_text,
        analyze_sentiments,
        classify_feedback,
        merge_and_save_data,
        push_to_github
    ]

    # Total steps in the process
    total_steps = len(tasks)
    progress_bar = tqdm(total=total_steps, desc="Overall Progress", unit="step")

    # Load new data from Google Sheet
    raw_data = tasks[0]()
    progress_bar.update(1)  # Update progress after each task

    # Load local data.csv to dataframe
    processed_data = tasks[1]()
    progress_bar.update(1)
    logger.info("Data.csv Loaded")

    # More steps follow...
    # Simulate each task with sleep and update progress bar
    for task in tasks[2:]:
        if task.__name__ == "process_new_data":
            data = raw_data[~raw_data.index.isin(processed_data.index)]
            if data.shape[0] != 0:

            else:
 
                print(f"{Fore.RED}[*] No New rows to add - terminated.")
                progress_bar.close()
                break
        result = task(data) if 'data' in task.__code__.co_varnames else task()
        progress_bar.update(1)

    # Close the progress bar at the end of all tasks
    if progress_bar.n != total_steps:
        progress_bar.update(total_steps - progress_bar.n)
    progress_bar.close()