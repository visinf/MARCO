"""AP-10K preprocessing script.

This script is adapted from the preprocessing pipeline of Geo-Aware-SC:
https://github.com/Junyi42/GeoAware-SC
"""

import os
import json
import shutil
import random
import numpy as np
import itertools

# Always resolve paths relative to the repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_and_merge_json(json_paths):
	merged_data = {'annotations': [], 'images': [], 'categories': []}
	for path in json_paths:
		with open(path, 'r') as file:
			data = json.load(file)
			merged_data['annotations'].extend(data['annotations'])
			merged_data['images'].extend(data['images'])
			if 'categories' in data and not merged_data['categories']:
				merged_data['categories'] = data['categories']
	return merged_data

def remove_duplicate_annotations(data):
	unique_image_ids = set()
	new_annotations = []
	for annotation in data['annotations']:
		if annotation['image_id'] not in unique_image_ids:
			unique_image_ids.add(annotation['image_id'])
			new_annotations.append(annotation)
	data['annotations'] = new_annotations
	return data

def merge_and_save_annotations(json_paths, base_dir):
	data = load_and_merge_json(json_paths)
	data = remove_duplicate_annotations(data)
	print(len(data['annotations']), "unique annotations after cleanup.")

	annotations = data["annotations"]
	images = data["images"]
	categories = data["categories"]

	images_dict = {image["id"]: image for image in images}
	categories_dict = {cat["id"]: cat for cat in categories}

	def pad_filename(filename):
		return filename.zfill(17)

	new_annotations = []
	for annotation in annotations:
		image_id = annotation["image_id"]
		category_id = annotation.get("category_id")
		if image_id in images_dict and category_id in categories_dict:
			new_annotation = {**annotation, **images_dict[image_id],
							  "name": categories_dict[category_id]["name"],
							  "supercategory": categories_dict[category_id]["supercategory"]}
			new_annotations.append(new_annotation)
			supercategory = new_annotation.get("supercategory")
			name = new_annotation.get("name")
			if supercategory and name:
				directory_path = os.path.join(base_dir, supercategory, name)
				os.makedirs(directory_path, exist_ok=True)
				file_name = pad_filename(str(new_annotation.get("id", "unknown")) + ".json")
				file_path = os.path.join(directory_path, file_name)
				with open(file_path, 'w') as file:
					json.dump(new_annotation, file, indent=4)
	print("All JSON files have been saved and renamed.")
	return new_annotations

def organize_images(new_annotations, img_folder, output_folder):
	id_to_category = {}
	for item in new_annotations:
		image_id = int(item.get("id", "unknown"))
		supercategory = item.get("supercategory")
		name = item.get("name")
		if supercategory and name:
			id_to_category[image_id] = os.path.join(supercategory, name)
	for img_file in os.listdir(img_folder):
		img_path = os.path.join(img_folder, img_file)
		if os.path.isfile(img_path):
			img_id = int(img_file.split(".")[0][7:])
			category_path = id_to_category.get(img_id)
			if category_path:
				target_directory = os.path.join(output_folder, category_path)
				os.makedirs(target_directory, exist_ok=True)
				target_path = os.path.join(target_directory, img_file)
				shutil.move(img_path, target_path)
	shutil.rmtree(img_folder)
	print("All images have been organized according to their respective categories.")

def split_json_files(json_files):
	length = len(json_files)
	test_size = min(30, length)
	eval_size = min(20, max(0, length - test_size))
	train_size = max(0, length - test_size - eval_size)
	train, eval, test = json_files[:train_size], json_files[train_size:train_size+eval_size], json_files[-test_size:]
	return train, eval, test

def filter_and_split_annotations(base_path, is_crowd_list_path):
	def load_list_from_file(file_path):
		with open(file_path, 'r') as file:
			return [line.strip() for line in file]
	def save_list_to_file(file_path, items):
		with open(file_path, 'w') as file:
			for item in items:
				file.write(f"{item}\n")
	is_crowd_list = load_list_from_file(is_crowd_list_path)
	for root, _, files in os.walk(base_path):
		if root.count(os.sep) == base_path.count(os.sep) + 2:
			json_list = [os.path.join(root, f) for f in files if f.endswith(".json")]
			filtered_json_list = []
			for json_file in json_list:
				with open(json_file, 'r') as f:
					json_data = json.load(f)
				if json_file.split('/')[-1].strip('.json') in is_crowd_list:
					json_data["is_crowd"] = 1
					with open(json_file, 'w') as f:
						json.dump(json_data, f)
				elif json_data["num_keypoints"] >= 3:
					filtered_json_list.append(json_file)
			train_json_list, eval_json_list, test_json_list = split_json_files(filtered_json_list)
			save_list_to_file(os.path.join(root, "train_filtered.txt"), train_json_list)
			save_list_to_file(os.path.join(root, "val_filtered.txt"), eval_json_list)
			save_list_to_file(os.path.join(root, "test_filtered.txt"), test_json_list)
	print("JSON file splitting complete.")

# ── Cell 4: Intra-species pair generation ──────────────────────────────────────
def generate_pairs(base_path, file_name, output_folder, N_multiplier=None):
	N_total_pairs = 0
	for root, dirs, files in os.walk(base_path):
		# Process only subdirectories that are two levels deep
		if root.count(os.sep) == base_path.count(os.sep) + 2:
			json_list = []
			with open(os.path.join(root, file_name), 'r') as file:
				json_list = [line.strip() for line in file]

			# For training, set N based on the length of json_list and a multiplier
			if N_multiplier is not None:
				N = 50 * len(json_list)  # Specific to training
			else:
				N = len(list(itertools.combinations(json_list, 2)))

			random.shuffle(json_list)
			all_possible_pairs = list(itertools.combinations(json_list, 2))
			possible_pairs = []

			for pair in all_possible_pairs:
				src_json_path, trg_json_path = pair
				with open(src_json_path) as f:
					src_data = json.load(f)
				with open(trg_json_path) as f:
					trg_data = json.load(f)
				src_kpt = np.array(src_data["keypoints"]).reshape(-1, 3)
				trg_kpt = np.array(trg_data["keypoints"]).reshape(-1, 3)
				mutual_vis = src_kpt[:, -1] / 2 * trg_kpt[:, -1] / 2
				if mutual_vis.sum() >= 3:
					possible_pairs.append(pair)

			N = min(N, len(possible_pairs))
			pairs_sampled = random.sample(possible_pairs, N) if N > 0 else []

			for pair in pairs_sampled:
				src_json_path, trg_json_path = pair
				src_json_name = os.path.basename(src_json_path).split(".")[0]
				trg_json_name = os.path.basename(trg_json_path).split(".")[0]
				category_name = os.path.basename(os.path.dirname(src_json_path))
				new_json_file_name = f"{src_json_name}-{trg_json_name}:{category_name}.json"
				if not os.path.exists(output_folder):
					os.makedirs(output_folder)
				new_json_path = os.path.join(output_folder, new_json_file_name)
				data = {"src_json_path": src_json_path, "trg_json_path": trg_json_path}
				with open(new_json_path, 'w') as f:
					json.dump(data, f, indent=4)
				N_total_pairs += 1
	print(f"Total {N_total_pairs} pairs generated for {file_name}")

# ── Cell 5: Cross-species pair generation ──────────────────────────────────────
def generate_cross_species_pairs(base_path, file_name, output_folder, N_pairs_per_combination):
	N_total_pairs = 0
	subfolder_path = {}
	for root, dirs, files in os.walk(base_path):
		if root.count(os.sep) == base_path.count(os.sep) + 1:
			subfolder_path[root] = []
			for subroot, subdirs, subfiles in os.walk(root):
				if subroot.count(os.sep) == root.count(os.sep) + 1:
					subfolder_path[root].append(subroot)

	for key, value in subfolder_path.items():
		if len(value) > 1:  # the family has more than one species
			total_cross_species_pairs = []
			species_combination = list(itertools.combinations(value, 2))
			for species_pair in species_combination:
				with open(os.path.join(species_pair[0], file_name), 'r') as train_file:
					train_json_list_1 = [line.strip() for line in train_file]
				with open(os.path.join(species_pair[1], file_name), 'r') as train_file:
					train_json_list_2 = [line.strip() for line in train_file]
				cross_species_pairs = list(itertools.product(train_json_list_1, train_json_list_2))
				for pair in cross_species_pairs:
					if random.random() > 0.5:
						pair = (pair[1], pair[0])
					total_cross_species_pairs.append(pair)

			possible_pairs = []
			for pair in total_cross_species_pairs:
				src_json_path, trg_json_path = pair
				with open(src_json_path) as f:
					src_data = json.load(f)
				with open(trg_json_path) as f:
					trg_data = json.load(f)
				src_kpt = np.array(src_data["keypoints"]).reshape(-1, 3)
				trg_kpt = np.array(trg_data["keypoints"]).reshape(-1, 3)
				src_vis = src_kpt[:, -1] / 2
				trg_vis = trg_kpt[:, -1] / 2
				mutual_vis = src_vis * trg_vis
				if mutual_vis.sum() >= 3:
					possible_pairs.append(pair)

			N = min(N_pairs_per_combination, len(possible_pairs))
			pairs_sampled = random.sample(possible_pairs, N) if N > 0 else []
			for pair in pairs_sampled:
				src_json_path, trg_json_path = pair
				src_json_name = os.path.basename(src_json_path).split(".")[0]
				trg_json_name = os.path.basename(trg_json_path).split(".")[0]
				category_name = os.path.basename(os.path.dirname(os.path.dirname(src_json_path)))
				new_json_file_name = f"{src_json_name}-{trg_json_name}:{category_name}.json"
				new_json_path = os.path.join(output_folder, new_json_file_name)
				if not os.path.exists(output_folder):
					os.makedirs(output_folder)
				data = {"src_json_path": src_json_path, "trg_json_path": trg_json_path}
				with open(new_json_path, 'w') as f:
					json.dump(data, f, indent=4)
			N_total_pairs += N
	print(f"Total {N_total_pairs} pairs for {file_name}")

# ── Cell 6: Cross-family pair generation ───────────────────────────────────────
def generate_cross_family_pairs(base_path, file_name, output_folder, N_pairs_per_combination):
	N_total_pairs = 0
	subfolder_path = {}
	for root, dirs, files in os.walk(base_path):
		if root.count(os.sep) == base_path.count(os.sep) + 1:
			subfolder_path[root] = []
			for subroot, subdirs, subfiles in os.walk(root):
				if subroot.count(os.sep) == root.count(os.sep) + 1:
					with open(os.path.join(subroot, file_name), 'r') as train_file:
						train_json_list = [line.strip() for line in train_file]
					subfolder_path[root].extend(train_json_list)

	families_combination = list(itertools.combinations(subfolder_path.keys(), 2))
	for family_pair in families_combination:
		family_1, family_2 = family_pair
		family_1_json_list = subfolder_path[family_1]
		family_2_json_list = subfolder_path[family_2]
		cross_family_pairs = list(itertools.product(family_1_json_list, family_2_json_list))
		possible_pairs = []
		for pair in cross_family_pairs:
			src_json_path, trg_json_path = pair
			with open(src_json_path) as f:
				src_data = json.load(f)
			with open(trg_json_path) as f:
				trg_data = json.load(f)
			src_kpt = np.array(src_data["keypoints"]).reshape(-1, 3)
			trg_kpt = np.array(trg_data["keypoints"]).reshape(-1, 3)
			mutual_vis = src_kpt[:, -1] / 2 * trg_kpt[:, -1] / 2
			if mutual_vis.sum() >= 3:
				possible_pairs.append(pair)
		N = min(N_pairs_per_combination, len(possible_pairs))
		pairs_sampled = random.sample(possible_pairs, N) if N > 0 else []
		for pair in pairs_sampled:
			src_json_path, trg_json_path = pair
			src_json_name = os.path.basename(src_json_path).split(".")[0]
			trg_json_name = os.path.basename(trg_json_path).split(".")[0]
			category_name = os.path.basename(os.path.dirname(os.path.dirname(src_json_path)))
			new_json_file_name = f"{src_json_name}-{trg_json_name}:all.json"
			new_json_path = os.path.join(output_folder, new_json_file_name)
			if not os.path.exists(output_folder):
				os.makedirs(output_folder)
			data = {"src_json_path": src_json_path, "trg_json_path": trg_json_path}
			with open(new_json_path, 'w') as f:
				json.dump(data, f, indent=4)
		N_total_pairs += N
	print(f"Total {N_total_pairs} pairs generated for {file_name}")

def main():

	# 1. Merge and save annotations
	json_paths = [
		os.path.join(REPO_ROOT, "data/ap-10k/annotations/ap10k-train-split1.json"),
		os.path.join(REPO_ROOT, "data/ap-10k/annotations/ap10k-test-split1.json"),
		os.path.join(REPO_ROOT, "data/ap-10k/annotations/ap10k-val-split1.json")
	]
	base_dir = os.path.join(REPO_ROOT, "data/ap-10k/ImageAnnotation")
	new_annotations = merge_and_save_annotations(json_paths, base_dir)

	# 2. Organize images
	img_folder = os.path.join(REPO_ROOT, "data/ap-10k/data")
	output_folder = os.path.join(REPO_ROOT, "data/ap-10k/JPEGImages")
	organize_images(new_annotations, img_folder, output_folder)

	# 3. Filter and split annotations
	filter_and_split_annotations(base_dir, os.path.join(REPO_ROOT, "data/ap-10k_is_crowd.txt"))

	# 4. Intra-species pairs
	pairann_dir = os.path.join(REPO_ROOT, "data/ap-10k/PairAnnotation")
	os.makedirs(pairann_dir, exist_ok=True)
	generate_pairs(base_dir, "train_filtered.txt", os.path.join(pairann_dir, "trn"), N_multiplier=50)
	generate_pairs(base_dir, "test_filtered.txt", os.path.join(pairann_dir, "test"))
	generate_pairs(base_dir, "val_filtered.txt", os.path.join(pairann_dir, "val"))

	# 5. Cross-species pairs
	generate_cross_species_pairs(base_dir, "val_filtered.txt", os.path.join(pairann_dir, "val_cross_species"), 400)
	generate_cross_species_pairs(base_dir, "test_filtered.txt", os.path.join(pairann_dir, "test_cross_species"), 900)

	# 6. Cross-family pairs
	generate_cross_family_pairs(base_dir, "test_filtered.txt", os.path.join(pairann_dir, "test_cross_family"), 30)
	generate_cross_family_pairs(base_dir, "val_filtered.txt", os.path.join(pairann_dir, "val_cross_family"), 20)

if __name__ == "__main__":
	main()
