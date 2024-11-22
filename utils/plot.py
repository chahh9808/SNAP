import numpy as np
import matplotlib.pyplot as plt
import random
import torch

# print out entropy/confidence statistics
# plots entropy vs confidence for (1) all data and (2) data used for adaptation
# class analysis, wdist analysis
def plot_prelim(all_list, mem_list, save_path_scatter, save_path_hist, save_path_classidx, save_path_wdist, save_path_time_wdist_train, save_path_time_wdist_test, save_path_time_confidence, save_path_time_entropy, text):
    all_entropies, all_confidences, all_wdists_test, all_wdists_train = zip(*all_list)
    mem_entropies, mem_confidences, mem_times, mem_classes, mem_wdists_test, mem_wdists_train = zip(*mem_list)

    # calculate stats
    avg_all_entropy = np.mean(all_entropies)
    avg_all_confidence = np.mean(all_confidences)
    avg_mem_entropy = np.mean(mem_entropies)
    avg_mem_confidence = np.mean(mem_confidences)
    avg_all_wdists_test = np.mean(all_wdists_test)
    avg_all_wdists_train = np.mean(all_wdists_train)
    avg_mem_wdists_test = np.mean(mem_wdists_test)
    avg_mem_wdists_train = np.mean(mem_wdists_train)

    # percentile calculation for hard threshold usage
    percentiles = np.percentile(all_wdists_test, [0, 20, 40, 50, 60, 80, 100])

    # Extract the median and thresholds for different percentile ranges
    median_value = percentiles[3]  # 50th percentile
    thresholds = {
        '0-20%': [percentiles[0], percentiles[1]],
        '20-40%': [percentiles[1], percentiles[2]],
        '40-60%': [percentiles[2], percentiles[4]],
        '60-80%': [percentiles[4], percentiles[5]],
        '80-100%': [percentiles[5], percentiles[6]],
    }

    # Output the results
    print(f"Median value (50th percentile): {median_value:.4f}")
    print(f"Thresholds for different percentiles:")
    for k, v in thresholds.items():
        formatted_values = [f"{val:.4f}" for val in v]
        print(f"{k}: {formatted_values}")

    with open(text, "a") as f:
        f.write(f"WDIST_test Median value (50th percentile): {median_value:.4f}")
        for k, v in thresholds.items():
            formatted_values = [f"{val:.4f}" for val in v]
            f.write(f"WDIST_test / {k}: {formatted_values}")

    # print stats
    print(f"Average entropy of all data: {avg_all_entropy:.2f}")
    print(f"Average confidence of all data: {avg_all_confidence:.2f}")
    print(f"Average entropy of adaptation data: {avg_mem_entropy:.2f}")
    print(f"Average confidence of adaptation data: {avg_mem_confidence:.2f}")
    print(f"Average WDIST_test of all data: {avg_all_wdists_test:.2f}")
    print(f"Average WDIST_train of all data: {avg_all_wdists_train:.2f}")
    print(f"Average WDIST_test of adaptation data: {avg_mem_wdists_test:.2f}")
    print(f"Average WDIST_train of adaptation data: {avg_mem_wdists_train:.2f}")

    with open(text, "a") as f:
        f.write(f"Average entropy of all data: {avg_all_entropy:.2f}" + "\n")
        f.write(f"Average confidence of all data: {avg_all_confidence:.2f}" + "\n")
        f.write(f"Average entropy of adaptation data: {avg_mem_entropy:.2f}" + "\n")
        f.write(f"Average confidence of adaptation data: {avg_mem_confidence:.2f}" + "\n")
        f.write(f"Average WDIST_test of all data: {avg_all_wdists_test:.2f}" + "\n")
        f.write(f"Average WDIST_train of all data: {avg_all_wdists_train:.2f}" + "\n")
        f.write(f"Average WDIST_test of adaptation data: {avg_mem_wdists_test:.2f}" + "\n")
        f.write(f"Average WDIST_train of adaptation data: {avg_mem_wdists_train:.2f}" + "\n")

    # print stats via adaptation period
    mem_data = list(zip(mem_entropies, mem_confidences, mem_times))
    mem_data_sorted = sorted(mem_data, key=lambda x: x[2])
    num_groups = 5
    group_size = len(mem_data_sorted) // num_groups

    for i in range(num_groups):
        group = mem_data_sorted[i*group_size:(i+1)*group_size]
        group_entropies, group_confidences, _ = zip(*group)
        avg_group_entropy = np.mean(group_entropies)
        avg_group_confidence = np.mean(group_confidences)
        print(f"Group {i+1} ({i*20}~{(i+1)*20}% time) - Average entropy: {avg_group_entropy:.2f}, Average confidence: {avg_group_confidence:.2f}")

    # plot scatter plot
    plt.figure(figsize=(10, 10))
    plt.scatter(all_entropies, all_confidences, color='dimgray', marker='x', label='All Data Points', alpha=0.3, s=10)
    scatter = plt.scatter(mem_entropies, mem_confidences, c=mem_times, cmap='viridis', marker='o', label='Data used for adaptation', alpha=0.5, s=10)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Time')

    plt.title('Entropy vs Confidence for Test Data)')
    plt.xlabel('Entropy')
    plt.ylabel('Confidence')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path_scatter)
    plt.close()

    # histogram for entropy
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)
    plt.hist(mem_entropies, bins=10, range=[0, 3], color='blue', alpha=0.7, weights=np.ones(len(mem_entropies)) / len(mem_entropies) * 100)
    plt.title('Histogram of Adaptation Data Entropy')
    plt.xlabel('Entropy')
    plt.ylabel('Percentage')
    plt.ylim(0, 100)

    # histogram for confidence
    plt.subplot(2, 1, 2)
    plt.hist(mem_confidences, bins=10, range=[0, 1], color='green', alpha=0.7, weights=np.ones(len(mem_confidences)) / len(mem_confidences) * 100)
    plt.title('Histogram of Adaptation Data Confidence')
    plt.xlabel('Confidence')
    plt.ylabel('Percentage')
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.savefig(save_path_hist)
    plt.close()

    # Histogram of class indices
    unique_classes, counts = np.unique(mem_classes, return_counts=True)
    plt.figure(figsize=(10, 5))
    plt.bar(unique_classes, counts, color='orange', alpha=0.7)
    plt.title('Histogram of Predicted Class Indices in Adaptation Data')
    plt.xlabel('Class Index')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(save_path_classidx)
    plt.close()

    # WDIST analysis
    plt.figure(figsize=(10, 6))
    plt.scatter(all_wdists_test, all_wdists_train, color='dimgray', alpha=0.2, label='All Data', s=10)
    plt.scatter(mem_wdists_test, mem_wdists_train, color='red', alpha=0.5, label='Adaptation Data', s=10)

    plt.title('Scatter Plot of Wasserstein Distances: Test vs Train')
    plt.xlabel('WDIST Test')
    plt.ylabel('WDIST Train')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path_wdist)
    plt.close()

    # WDIST analysis throughout adaptation

    # time vs wdist_train
    plt.figure(figsize=(10, 6))
    plt.scatter(mem_times, mem_wdists_train, c='blue', alpha=0.1)
    
    plt.title('WDIST Train (of memory samples) Throughout Adaptation')
    plt.xlabel('Time')
    plt.ylabel('WDIST Train')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path_time_wdist_train)
    plt.close()

    # time vs wdist_test
    plt.figure(figsize=(10, 6))
    plt.scatter(mem_times, mem_wdists_test, c='blue', alpha=0.1)
    
    plt.title('WDIST test (of memory samples) Throughout Adaptation')
    plt.xlabel('Time')
    plt.ylabel('WDIST Test')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path_time_wdist_test)
    plt.close()

    # time vs confidence
    plt.figure(figsize=(10, 6))
    plt.scatter(mem_times, mem_confidences, c='blue', alpha=0.1)
    
    plt.title('Confidence (of memory samples) Throughout Adaptation')
    plt.xlabel('Time')
    plt.ylabel('Confidence')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path_time_confidence)
    plt.close()

    # time vs entropy
    plt.figure(figsize=(10, 6))
    plt.scatter(mem_times, mem_entropies, c='blue', alpha=0.1)
    
    plt.title('Entropy (of memory samples) Throughout Adaptation')
    plt.xlabel('Time')
    plt.ylabel('Entropy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path_time_entropy)
    plt.close()


# label flip analysis
def plot_label_flip(data_list, filename="prelim_label_flip.png"):
    # Sample 10% of the data for the scatter plot
    sample_size = int(0.1 * len(data_list))
    sampled_data = random.sample(data_list, sample_size)

    # 4 different cases of label flips (and non-flips)
    entropy_correct_source_incorrect = [x[0] for x in sampled_data if x[2] and not x[3]]
    confidence_correct_source_incorrect = [x[1] for x in sampled_data if x[2] and not x[3]]
    entropy_incorrect_source_incorrect = [x[0] for x in sampled_data if not x[2] and not x[3]]
    confidence_incorrect_source_incorrect = [x[1] for x in sampled_data if not x[2] and not x[3]]
    entropy_correct_source_correct = [x[0] for x in sampled_data if x[2] and x[3]]
    confidence_correct_source_correct = [x[1] for x in sampled_data if x[2] and x[3]]
    entropy_incorrect_source_correct = [x[0] for x in sampled_data if not x[2] and x[3]]
    confidence_incorrect_source_correct = [x[1] for x in sampled_data if not x[2] and x[3]]

    plt.figure(figsize=(15, 10))

    # Scatter plot
    plt.subplot(3, 1, 1)
    plt.scatter(entropy_correct_source_incorrect, confidence_correct_source_incorrect, alpha=0.5, s=10, label='Correct but Source Incorrect', color='green')
    plt.scatter(entropy_incorrect_source_incorrect, confidence_incorrect_source_incorrect, alpha=0.5, s=10, label='Incorrect and Source Incorrect', color='red')
    plt.scatter(entropy_correct_source_correct, confidence_correct_source_correct, alpha=0.5, s=10, label='Correct and Source Correct', color='blue')
    plt.scatter(entropy_incorrect_source_correct, confidence_incorrect_source_correct, alpha=0.5, s=10, label='Incorrect but Source Correct', color='orange')
    plt.title('Label flip (Entropy vs Confidence)')
    plt.xlabel('Entropy')
    plt.ylabel('Confidence')
    plt.grid(True)
    plt.xlim(0, max([x[0] for x in sampled_data]) * 1.1)
    plt.ylim(0, max([x[1] for x in sampled_data]) * 1.1)
    plt.legend()

    # Histograms
    plt.subplot(3, 1, 2)
    bins = np.linspace(0, max([x[0] for x in data_list]) * 1.1, 10)
    
    counts_correct_source_incorrect, _ = np.histogram(entropy_correct_source_incorrect, bins)
    counts_incorrect_source_incorrect, _ = np.histogram(entropy_incorrect_source_incorrect, bins)
    counts_correct_source_correct, _ = np.histogram(entropy_correct_source_correct, bins)
    counts_incorrect_source_correct, _ = np.histogram(entropy_incorrect_source_correct, bins)
    
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    width = (bins[1] - bins[0]) / 5
    
    plt.bar(bin_centers - 1.5 * width, counts_correct_source_incorrect, width=width, label='Correct but Source Incorrect', color='green')
    plt.bar(bin_centers - 0.5 * width, counts_incorrect_source_incorrect, width=width, label='Incorrect and Source Incorrect', color='red')
    plt.bar(bin_centers + 0.5 * width, counts_correct_source_correct, width=width, label='Correct and Source Correct', color='blue')
    plt.bar(bin_centers + 1.5 * width, counts_incorrect_source_correct, width=width, label='Incorrect but Source Correct', color='orange')
    
    plt.title('Label Flips and Entropy Values')
    plt.xlabel('Entropy')
    plt.ylabel('Count')
    plt.ylim(0, 500)
    plt.legend()

    plt.subplot(3, 1, 3)
    bins = np.linspace(0, max([x[1] for x in data_list]) * 1.1, 10)
    
    counts_correct_source_incorrect, _ = np.histogram(confidence_correct_source_incorrect, bins)
    counts_incorrect_source_incorrect, _ = np.histogram(confidence_incorrect_source_incorrect, bins)
    counts_correct_source_correct, _ = np.histogram(confidence_correct_source_correct, bins)
    counts_incorrect_source_correct, _ = np.histogram(confidence_incorrect_source_correct, bins)
    
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    width = (bins[1] - bins[0]) / 5
    
    plt.bar(bin_centers - 1.5 * width, counts_correct_source_incorrect, width=width, label='Correct but Source Incorrect', color='green')
    plt.bar(bin_centers - 0.5 * width, counts_incorrect_source_incorrect, width=width, label='Incorrect and Source Incorrect', color='red')
    plt.bar(bin_centers + 0.5 * width, counts_correct_source_correct, width=width, label='Correct and Source Correct', color='blue')
    plt.bar(bin_centers + 1.5 * width, counts_incorrect_source_correct, width=width, label='Incorrect but Source Correct', color='orange')
    
    plt.title('Label Flips and Confidence Values')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.ylim(0, 500)
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# dataset analysis
def plot_data(data_list, filename):
    entropies, confidences, wdists_test, wdists_train = zip(*data_list)

    # calculate stats
    avg_entropy = np.mean(entropies)
    median_entropy = np.median(entropies)
    avg_confidence = np.mean(confidences)
    median_confidence = np.median(confidences)

    # print stats
    print(f"Mean entropy: {avg_entropy:.2f}")
    print(f"Median entropy: {median_entropy:.2f}")
    print(f"Mean confidence: {avg_confidence:.2f}")
    print(f"Median confidence: {median_confidence:.2f}")

    # plot histograms
    plt.figure(figsize=(10, 10))

    # histogram for entropy
    plt.subplot(2, 1, 1)
    plt.hist(entropies, bins=30, range=[0, max(entropies) * 1.1], color='blue', alpha=0.7, weights=np.ones(len(entropies)) / len(entropies) * 100)
    plt.title('Histogram of Entropy of all data')
    plt.xlabel('Entropy')
    plt.ylabel('Percentage')
    plt.ylim(0, 100)

    # histogram for confidence
    plt.subplot(2, 1, 2)
    plt.hist(confidences, bins=30, range=[0, 1], color='green', alpha=0.7, weights=np.ones(len(confidences)) / len(confidences) * 100)
    plt.title('Histogram of Confidence of all data')
    plt.xlabel('Confidence')
    plt.ylabel('Percentage')
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_confidence_histogram(conf_list, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.hist(conf_list, bins=1000, color='blue', alpha=0.7, edgecolor='black')
    
    plt.title('Predicted Confidence of GT correct classes')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')

    plt.tight_layout() 
    plt.savefig(save_path)
    plt.close()

def plot_sample(sample_number_list, filename):
    # Extract time steps and sample counts from the sample_number_list
    time_steps, sample_counts = zip(*sample_number_list)

    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, sample_counts, marker='o', linestyle='', color='b', alpha=0.1)
    plt.title('Sample Statistics Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Number of Samples Satisfying Threshold')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_avg_accuracy(acc_list, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(acc_list)
    plt.xlabel('Batch Number')
    plt.ylabel('Accuracy (%)')
    plt.title('Average Accuracy througout Adaptation')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_bn_stats_wass(bn_stats_list, filename_test, filename_train):
    entropies = [stat[0] for stat in bn_stats_list]
    wasserstein_distances_test = [stat[1] for stat in bn_stats_list]
    wasserstein_distances_train = [stat[2] for stat in bn_stats_list]

    # wdist_test
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.scatter(entropies, wasserstein_distances_test, color='purple', alpha=0.6)
    plt.title('Entropy vs WDIST_Test')
    plt.xlabel('Entropy')
    plt.ylabel('WDIST_Test')

    plt.subplot(2, 1, 2)
    plt.hist(wasserstein_distances_test, bins=20, color='purple', alpha=0.7)
    plt.title('Histogram of WDIST_Test')
    plt.xlabel('WDIST_Test')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(filename_test)
    plt.close()

    # wdist_train
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.scatter(entropies, wasserstein_distances_train, color='purple', alpha=0.6)
    plt.title('Entropy vs WDIST_Train')
    plt.xlabel('Entropy')
    plt.ylabel('WDIST_Train')

    plt.subplot(2, 1, 2)
    plt.hist(wasserstein_distances_train, bins=20, color='purple', alpha=0.7)
    plt.title('Histogram of WDIST_Train')
    plt.xlabel('WDIST_Train')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(filename_train)
    plt.close()

def plot_wass_correct(wass_correctness_list, filename_test, filename_train, text):
    # Separate Wasserstein distances into two lists: correct and incorrect predictions
    correct_distances_test = [dist_test for dist_test, dist_train, correct in wass_correctness_list if correct]
    incorrect_distances_test = [dist_test for dist_test, dist_train, correct in wass_correctness_list if not correct]

    correct_distances_train = [dist_train for dist_test, dist_train, correct in wass_correctness_list if correct]
    incorrect_distances_train = [dist_train for dist_test, dist_train, correct in wass_correctness_list if not correct]

    with open(text, "a") as f:
        f.write(f"WDIST_test of correct predictions: {np.mean(correct_distances_test):.2f}" + "\n")
        f.write(f"WDIST_test of incorrect predictions: {np.mean(incorrect_distances_test):.2f}" + "\n")
        f.write(f"WDIST_train of correct predictions: {np.mean(correct_distances_train):.2f}" + "\n")
        f.write(f"WDIST_train of incorrect predictions: {np.mean(incorrect_distances_train):.2f}" + "\n")

    # wdist_test
    plt.figure(figsize=(10, 6))
    plt.hist(correct_distances_test, bins=30, alpha=0.7, label='Correct Predictions', color='blue')
    plt.hist(incorrect_distances_test, bins=30, alpha=0.7, label='Incorrect Predictions', color='red')
    plt.title('WDIST_Test Histogram for Correct and Incorrect Predictions')
    plt.xlabel('WDIST_Test')
    plt.ylabel('Frequency')
    plt.ylim(0, max(plt.gca().get_ylim()[1], plt.gca().get_ylim()[1]))
    plt.legend(loc='upper right')
    plt.savefig(filename_test)
    plt.close()

    # wdist_train
    plt.figure(figsize=(10, 6))
    plt.hist(correct_distances_train, bins=30, alpha=0.7, label='Correct Predictions', color='blue')
    plt.hist(incorrect_distances_train, bins=30, alpha=0.7, label='Incorrect Predictions', color='red')
    plt.title('WDIST_Train Histogram for Correct and Incorrect Predictions')
    plt.xlabel('WDIST_Train')
    plt.ylabel('Frequency')
    plt.ylim(0, max(plt.gca().get_ylim()[1], plt.gca().get_ylim()[1]))
    plt.legend(loc='upper right')
    plt.savefig(filename_train)
    plt.close()

def plot_conf_correct(conf_correctness_list, filename, text):
    # Separate confidence into two lists: correct and incorrect predictions
    correct_confs = [conf.cpu().numpy() if isinstance(conf, torch.Tensor) else conf for conf, correct in conf_correctness_list if correct]
    incorrect_confs = [conf.cpu().numpy() if isinstance(conf, torch.Tensor) else conf for conf, correct in conf_correctness_list if not correct]

    percent_correct = len(correct_confs) / (len(correct_confs) + len(incorrect_confs))

    with open(text, "a") as f:
        f.write(f"Pseudo-label accuracy of all data: {percent_correct:.4f}" + "\n")

    print(f"Pseudo-label accuracy of all data: {percent_correct:.4f}")
    
    # Plot histograms
    plt.figure(figsize=(10, 6))
    
    # Plot for correct samples
    plt.hist(correct_confs, bins=30, alpha=0.7, label='Correct Predictions', color='blue')
    
    # Plot for incorrect samples
    plt.hist(incorrect_confs, bins=30, alpha=0.7, label='Incorrect Predictions', color='red')
    
    # Add titles and labels
    plt.title('Confidence Histogram for Correct and Incorrect Predictions')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    
    # Ensure both histograms share the same y-axis scale
    plt.ylim(0, max(plt.gca().get_ylim()[1], plt.gca().get_ylim()[1]))
    
    # Add legend
    plt.legend(loc='upper right')
    plt.savefig(filename)
    plt.close()
