import os
import Levenshtein
import time
from clustering.clustering import (
    create_clusters,
    create_ground_truth_synonym_clusters,
    postprocess_by_region,
)
from clustering.accuracy_test import (
    clustering_accuracy,
    average_inter_cluster_similarity,
    average_intra_cluster_similarity,
    clean_ground_truth_clusters,
)
from string_cleaning import clean_name
from interface import display_menu
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import json
import random


def is_standard_variant(
    row_name, standard_name, cleaned_row_name, cleaned_standard_name
):
    # Check if row name matches the standard name
    if row_name == standard_name or cleaned_row_name in cleaned_standard_name:
        return True

    # Check if row name is similar to the standard name
    distance_threshold = len(cleaned_row_name) * 0.3
    distance_to_standard_name = Levenshtein.distance(
        cleaned_row_name, cleaned_standard_name
    )
    if distance_to_standard_name <= distance_threshold:
        return True
    else:
        return False


def load_dataset(file_path):
    # Verify if the file exists
    if not os.path.isfile(file_path):
        return None

    # Load the file into a dataframe, according to its file type
    if file_path.endswith(".csv"):
        try:
            df = pd.read_csv(
                file_path,
                usecols=["name", "identification_number"],
                dtype={"name": str, "identification_number": str},
            )
        except (pd.errors.ParserError, ValueError):
            df = pd.read_csv(
                file_path,
                delimiter=";",
                usecols=["name", "identification_number"],
                dtype={"name": str, "identification_number": str},
            )
    elif file_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(
            file_path,
            usecols=["name", "identification_number"],
            dtype={"name": str, "identification_number": str},
        )
    else:
        print(f"[ERROR] Unsupported file type: {file_path}")
        return None

    # Return the data frame
    return df


def main():
    from colorama import Fore, Style, init

    init(autoreset=True)  # Initialize colorama for automatic reset of styles

    standard_datasets = set()
    datasets_to_standardize = set()
    uniques_names_to_standardize = set()
    evaluate_clusters = False

    # Display the menu and handle user input
    standard_datasets, datasets_to_standardize = display_menu(
        standard_datasets, datasets_to_standardize
    )
    choice = (
        input(
            Fore.CYAN
            + Style.BRIGHT
            + "\nDo you wish to make tests to the clusters? [y/N]: "
        )
        .strip()
        .lower()
    )
    if choice == "y":
        print(Fore.GREEN + "✔ Cluster evaluation activated!")
        evaluate_clusters = True

    #######################################################################################################
    # Header
    print(Fore.CYAN + Style.BRIGHT + "\n" + "=" * 60)
    print(Fore.CYAN + Style.BRIGHT + "{:^60}".format("SETUP"))
    print(Fore.CYAN + Style.BRIGHT + "=" * 60 + Style.RESET_ALL)

    # Start timer
    start_time = time.time()

    # Initialization
    ground_truth_synonym_map = defaultdict(lambda: defaultdict(list))
    id_to_name_map = {}
    repeated_ids = set()

    # Process standard datasets
    if standard_datasets:
        print(
            Fore.YELLOW + "\n[STEP] Processing Standard Datasets..." + Style.RESET_ALL
        )
        for dataset_path in standard_datasets:
            print(Fore.BLUE + f"[LOAD] Dataset: {dataset_path}" + Style.RESET_ALL)
            df = load_dataset(dataset_path)
            for index in range(len(df)):
                standard_name = str(df.at[index, "name"])
                cleaned_name = clean_name(standard_name)
                identification_number = str(df.at[index, "identification_number"])

                # Fill synonym map
                if (
                    identification_number
                    not in ground_truth_synonym_map[cleaned_name][cleaned_name]
                ):
                    ground_truth_synonym_map[cleaned_name][cleaned_name].append(
                        identification_number
                    )

                # Handle repeated IDs and id to name map
                if identification_number in repeated_ids:
                    continue
                elif identification_number in id_to_name_map:
                    repeated_ids.add(identification_number)
                    id_to_name_map.pop(identification_number)
                else:
                    id_to_name_map[identification_number] = standard_name
        print(
            Fore.GREEN
            + "[INFO] Standard datasets processed successfully!"
            + Style.RESET_ALL
        )

    # Process datasets to standardize
    for dataset_path in datasets_to_standardize:
        print(
            Fore.YELLOW
            + f"\n[STEP] Processing Dataset to Standardize: {dataset_path}"
            + Style.RESET_ALL
        )
        df = load_dataset(dataset_path)
        corrected_count = 0
        repeated_ids_skip = 0

        for row in tqdm(df.itertuples(index=False), desc="Processing Rows", unit="row"):
            row_name = str(row.name)
            identification_number = str(row.identification_number)
            cleaned_row_name = clean_name(row_name)
            uniques_names_to_standardize.add(cleaned_row_name)

            if identification_number in repeated_ids:
                repeated_ids_skip += 1
                continue

            if row_name in ground_truth_synonym_map:
                corrected_count += 1
                continue

            if identification_number in id_to_name_map:
                standard_name = id_to_name_map[identification_number]
                cleaned_standard_name = clean_name(standard_name)

                if is_standard_variant(
                    row_name, standard_name, cleaned_row_name, cleaned_standard_name
                ):
                    entry = ground_truth_synonym_map.setdefault(cleaned_row_name, {})
                    ids = entry.setdefault(cleaned_standard_name, [])
                    if identification_number not in ids:
                        ids.append(identification_number)
                    corrected_count += 1
                else:
                    continue
            else:
                continue

        # Summary
        print(Fore.CYAN + "\n" + "-" * 60)
        print(Fore.CYAN + "Summary of Synonym Map Correspondence:" + Style.RESET_ALL)
        print(Fore.GREEN + f"  - Total Rows Processed: {len(df)}" + Style.RESET_ALL)
        print(
            Fore.YELLOW
            + f"  - Multiple Identification Number Count: {repeated_ids_skip}"
            + Style.RESET_ALL
        )
        print(
            Fore.GREEN
            + f"  - Rows with Synonym Map Correspondence: {corrected_count}"
            + Style.RESET_ALL
        )
        print(
            Fore.CYAN
            + f"  - Percentage: {(corrected_count / len(df)) * 100:.2f}%"
            + Style.RESET_ALL
        )
        print(Fore.CYAN + "-" * 60 + Style.RESET_ALL)

    # Convert synonym map to cluster
    synonym_cluster = create_ground_truth_synonym_clusters(ground_truth_synonym_map)

    # Execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    # Final time report
    print(
        Fore.GREEN
        + Style.BRIGHT
        + f"\n[TIME] Execution time: {elapsed_time:.2f} seconds --> {hours}h {minutes}m {seconds}s"
        + Style.RESET_ALL
    )
    #######################################################################################################

    #######################################################################################################
    # Header
    print(Fore.CYAN + Style.BRIGHT + "\n" + "=" * 60)
    print(Fore.CYAN + Style.BRIGHT + "{:^60}".format("CREATING CLUSTERS"))
    print(Fore.CYAN + Style.BRIGHT + "=" * 60 + Style.RESET_ALL)

    # Start timer
    start_time = time.time()

    # Cluster creation
    clusters = create_clusters(
        names=uniques_names_to_standardize,
        similarity_threshold=0.81,
        base_clusters=synonym_cluster,
    )

    # Initial cluster info
    print(
        Fore.YELLOW
        + "\n[INFO] Number of clusters: "
        + Style.RESET_ALL
        + f"{len(clusters)}"
    )
    print(
        Fore.YELLOW
        + "[INFO] Unique names in clusters: "
        + Style.RESET_ALL
        + f"{len(set(name for cluster in clusters for name in cluster))}"
    )

    # Post-processing
    print(
        Fore.MAGENTA
        + "\n[STEP] Post-processing clusters by region..."
        + Style.RESET_ALL
    )
    clusters = postprocess_by_region(clusters)

    # Post-processed cluster info
    print(
        Fore.YELLOW
        + "\n[INFO] Number of clusters after post-processing: "
        + Style.RESET_ALL
        + f"{len(clusters)}"
    )
    print(
        Fore.YELLOW
        + "[INFO] Unique names in clusters: "
        + Style.RESET_ALL
        + f"{len(set(name for cluster in clusters for name in cluster))}"
    )

    print(Fore.YELLOW + "\n[STEP] Calculating Cluster Similarity..." + Style.RESET_ALL)
    intra_sim = average_intra_cluster_similarity(clusters)
    inter_sim = average_inter_cluster_similarity(clusters)
    print(
        Fore.GREEN
        + f"\n  - Average Intra-Cluster Similarity: {intra_sim:.2f}"
        + Style.RESET_ALL
    )
    print(
        Fore.GREEN
        + f"  - Average Inter-Cluster Similarity: {inter_sim:.2f}"
        + Style.RESET_ALL
    )

    # Execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    # Final time report
    print(
        Fore.GREEN
        + Style.BRIGHT
        + f"\n[TIME] Execution time: {elapsed_time:.2f} seconds --> {hours}h {minutes}m {seconds}s"
        + Style.RESET_ALL
    )
    #######################################################################################################

    #######################################################################################################
    if evaluate_clusters:
        # Header
        print(Fore.CYAN + Style.BRIGHT + "\n" + "=" * 60)
        print(Fore.CYAN + Style.BRIGHT + "{:^60}".format("EVALUATE CLUSTERS"))
        print(Fore.CYAN + Style.BRIGHT + "=" * 60 + Style.RESET_ALL)

        # Start timer
        start_time = time.time()

        print(
            Fore.YELLOW + "\n[STEP] Cleaning Ground Truth Clusters..." + Style.RESET_ALL
        )
        cleaned_ground_truth_clusters = clean_ground_truth_clusters(synonym_cluster)
        cleaned_ground_truth_clusters_names = set(
            name for cluster in cleaned_ground_truth_clusters for name in cluster
        )

        print(Fore.CYAN + "\n" + "-" * 60 + Style.RESET_ALL)
        print(
            Fore.MAGENTA
            + "\n[VERSION 1] Predicted Clusters Without Base Clusters"
            + Style.RESET_ALL
        )

        print(Fore.BLUE + "\n[INFO] Creating Predicted Clusters..." + Style.RESET_ALL)
        predicted_clusters = create_clusters(
            names=cleaned_ground_truth_clusters_names, similarity_threshold=0.76
        )
        predicted_clusters = postprocess_by_region(predicted_clusters)

        print(Fore.BLUE + "\n[INFO] Evaluating Accuracy..." + Style.RESET_ALL)
        predicted_cluster_evaluation = clustering_accuracy(
            synonym_cluster, predicted_clusters
        )

        print(Fore.GREEN + "[RESULT] Predicted Clusters Evaluation:" + Style.RESET_ALL)
        print(
            Fore.GREEN
            + f"  - Precision: {int(predicted_cluster_evaluation['precision'] * 100)}%"
            + Style.RESET_ALL
        )
        print(
            Fore.YELLOW
            + f"  - Recall:    {int(predicted_cluster_evaluation['recall'] * 100)}%"
            + Style.RESET_ALL
        )
        print(
            Fore.CYAN
            + f"  - F1 Score:  {int(predicted_cluster_evaluation['f1_score'] * 100)}%"
            + Style.RESET_ALL
        )

        print(Fore.CYAN + "\n" + "-" * 60 + Style.RESET_ALL)
        print(
            Fore.MAGENTA
            + "\n[VERSION 2] Predicted Clusters with base clusters (50% of the synonym clusters)"
            + Style.RESET_ALL
        )
        base_percent = 0.5
        synonym_base_clusters = synonym_cluster.copy()
        names_to_add = set()
        for i in range(int(len(synonym_cluster) * base_percent)):
            random_index = random.randint(0, len(synonym_base_clusters) - 1)
            removed_cluster = synonym_base_clusters.pop(random_index)
            names_to_add.update(set(removed_cluster))

        print(Fore.BLUE + "\n[INFO] Creating Predicted Clusters..." + Style.RESET_ALL)
        predicted_clusters_with_base_clusters = create_clusters(
            names=names_to_add,
            similarity_threshold=0.76,
            base_clusters=synonym_base_clusters,
        )
        predicted_clusters_with_base_clusters = postprocess_by_region(
            predicted_clusters_with_base_clusters
        )

        print(Fore.BLUE + "\n[INFO] Evaluating Accuracy..." + Style.RESET_ALL)
        predicted_cluster_evaluation = clustering_accuracy(
            synonym_cluster, predicted_clusters_with_base_clusters
        )

        print(Fore.GREEN + "[RESULT] Predicted Clusters Evaluation:" + Style.RESET_ALL)
        print(
            Fore.GREEN
            + f"  - Precision: {int(predicted_cluster_evaluation['precision'] * 100)}%"
            + Style.RESET_ALL
        )
        print(
            Fore.YELLOW
            + f"  - Recall:    {int(predicted_cluster_evaluation['recall'] * 100)}%"
            + Style.RESET_ALL
        )
        print(
            Fore.CYAN
            + f"  - F1 Score:  {int(predicted_cluster_evaluation['f1_score'] * 100)}%"
            + Style.RESET_ALL
        )

        # Execution time
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        # Final time report
        print(
            Fore.GREEN
            + Style.BRIGHT
            + f"\n[TIME] Execution time: {elapsed_time:.2f} seconds --> {hours}h {minutes}m {seconds}s"
            + Style.RESET_ALL
        )
    #######################################################################################################

    #######################################################################################################
    # Header
    print(Fore.CYAN + Style.BRIGHT + "\n" + "=" * 60)
    print(Fore.CYAN + Style.BRIGHT + "{:^60}".format("CREATE FINAL SYNONYM MAP"))
    print(Fore.CYAN + Style.BRIGHT + "=" * 60 + Style.RESET_ALL)

    # Start timer
    print(Fore.YELLOW + "\n[STEP] Creating final synonym map..." + Style.RESET_ALL)

    # Start timing the process
    start_time = time.time()

    # Build the final synonym map: associate each name with the first name in its cluster
    final_version_synonym_map = {}
    for cluster in clusters:
        for name in cluster:
            final_version_synonym_map[name] = cluster[0]

    # Calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    # Print the execution time
    print(
        Fore.GREEN
        + Style.BRIGHT
        + f"\n[TIME] Execution time: {elapsed_time:.2f} seconds --> {hours}h {minutes}m {seconds}s"
        + Style.RESET_ALL
    )
    #######################################################################################################

    #######################################################################################################
    # Header
    print(Fore.CYAN + Style.BRIGHT + "\n" + "=" * 60)
    print(
        Fore.CYAN
        + Style.BRIGHT
        + "{:^60}".format("CORRECTING DATASETS WITH SYNONYM MAP")
    )
    print(Fore.CYAN + Style.BRIGHT + "=" * 60 + Style.RESET_ALL)

    # Start timer
    start_time = time.time()

    # Loop through all datasets to be standardized
    for dataset_path in datasets_to_standardize:
        # Load the dataset
        df = load_dataset(dataset_path)

        # Inform which file is being processed
        print(
            Fore.YELLOW + f"\n[INFO] Processing file: {dataset_path}" + Style.RESET_ALL
        )

        standard_not_found = 0
        new_standard_names = []

        # Go through each row and standardize the name
        for row in tqdm(df.itertuples(index=False), desc="Processing Rows", unit="row"):
            cleaned_row_name = clean_name(str(row.name))

            # If a standardized version exists, use it
            if cleaned_row_name in final_version_synonym_map:
                new_standard_names.append(final_version_synonym_map[cleaned_row_name])
            else:
                new_standard_names.append("None")
                standard_not_found += 1

        # Add new standardized names to the dataframe
        df["new_standard_name"] = new_standard_names

        # Print number of unmatched names
        print(
            Fore.MAGENTA
            + f"[WARNING] No standard found for {standard_not_found} entries."
            + Style.RESET_ALL
        )

        # Create output file path
        dir_name = os.path.dirname(dataset_path)
        base_name = os.path.basename(dataset_path)
        name_without_ext, ext = os.path.splitext(base_name)
        new_filename = f"{name_without_ext}-standardized{ext}"
        new_file_path = os.path.join(dir_name, new_filename)

        # Save the standardized file
        df.to_csv(new_file_path, index=False)
        print(
            Fore.GREEN
            + f"[SAVED] Standardized dataset saved to: {new_file_path}"
            + Style.RESET_ALL
        )

    # Print execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    print(
        Fore.GREEN
        + Style.BRIGHT
        + f"\n[TIME] Execution time: {elapsed_time:.2f} seconds --> {hours}h {minutes}m {seconds}s"
        + Style.RESET_ALL
    )
    #######################################################################################################

    # End of script message
    print(Fore.CYAN + Style.BRIGHT + "\n" + "#" * 60)
    print(Fore.CYAN + Style.BRIGHT + "{:^60}".format("END OF SCRIPT"))
    print(Fore.CYAN + Style.BRIGHT + "#" * 60 + Style.RESET_ALL)


if __name__ == "__main__":
    main()

