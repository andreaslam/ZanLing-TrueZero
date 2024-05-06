import matplotlib.pyplot as plt
import io
import re


def extract_data_fmt(input_string):
    # example entry
    # 1710608194100142900 1710608194400395200 executor_0 waiting_for_batch

    return re.findall(r"\d+\s\d+\s\S+\s\S+\s", input_string)


def read_tasks_from_file(file_path):
    with io.open(file_path, "r", encoding="utf-16le") as f:
        tasks_data = f.read()
    return extract_data_fmt(tasks_data)


def process_tasks(tasks):
    tasks = [t.replace("\n", "") for t in tasks]
    tasks = [t.split(" ") for t in tasks]
    list_of_times = [t[0] for t in tasks]
    list_of_times.sort()
    init_time = int(list_of_times[0])
    return [
        [
            (int(sublist[0]) - init_time) / 1e9,
            (int(sublist[1]) - init_time) / 1e9,
            sublist[2],
            sublist[3],
        ]
        for sublist in tasks
    ]


def plot_schedule(tasks):
    task_desc_colors = {}  # Store colours for each task description
    threads = sorted(set(task[2] for task in tasks))
    thread_positions = {thread: index for index, thread in enumerate(threads)}

    plt.figure(figsize=(10, 6))

    for task in sorted(tasks, key=lambda x: x[2]):
        thread_name = task[2]
        thread_index = thread_positions[thread_name]
        overlapping_tasks = [
            t
            for t in tasks
            if t[2] == thread_name and t[0] < task[1] and t[1] > task[0]
        ]
        height = 0.5  # Adjust height here for the vertical space between tasks
        num_overlaps = len(overlapping_tasks)
        if num_overlaps > 1:
            for i, overlap_task in enumerate(overlapping_tasks, start=1):
                color = task_desc_colors.setdefault(
                    overlap_task[3], plt.cm.tab10(len(task_desc_colors) % 10)
                )
                plt.barh(
                    thread_index + (i * height),  # Adjusted height calculation
                    overlap_task[1] - overlap_task[0],
                    left=overlap_task[0],
                    height=height,
                    color=color,
                    edgecolor="black",
                    linewidth=0,
                )
        else:
            color = task_desc_colors.setdefault(
                task[3], plt.cm.tab10(len(task_desc_colors) % 10)
            )

            plt.barh(
                thread_index,
                task[1] - task[0],
                left=task[0],
                height=height,
                color=color,
                label=task[3],
            )

    # Legend creation
    legend_labels = set(task[3] for task in tasks)
    legend_handles = [
        plt.bar(0, 0, color=task_desc_colors[label], label=label)
        for label in legend_labels
    ]
    plt.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.05, 1))

    plt.yticks(
        list(thread_positions.values()),
        [f"Thread {thread_name}" for thread_name in threads],
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Thread")
    plt.title("Thread Schedule")
    plt.grid(True)

    # Set x and y axis limits to start from origin
    plt.xlim(0, max(task[1] for task in tasks))  # Setting x-axis limit
    plt.ylim(-0.5, len(threads) - 0.5)  # Setting y-axis limit

    plt.tight_layout()
    plt.savefig("plt.png")
    # plt.show()


# Example usage:
tasks = process_tasks(read_tasks_from_file("logs.txt"))
plot_schedule(tasks)
