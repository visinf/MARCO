#!/bin/bash

# Prepare AP-10K-derived animal-body categories for the MP-100 benchmark.
#
# This script creates symbolic links from the processed AP-10K image folders
# to the category names used by the MP-100 benchmark.
#
# Usage:
#   ./process_ap10k_for_mp100.sh BASE_DIR [mp100_root]

set -e

if [[ -z "$1" ]]; then
	echo "Usage: $0 BASE_DIR [mp100_root]"
	echo "  BASE_DIR: Path to AP-10K JPEGImages (e.g. data/ap-10k/JPEGImages)"
	echo "  mp100_root: (optional) Path to mp100 root (default: data/mp100_all/mp100)"
	exit 1
fi

BASE_DIR="$1"
MP100_ROOT="${2:-data/mp100_all/mp100}"

# Declare mapping: "source_path|symlink_name"
declare -a MAPPINGS=(
	"Bovidae/antelope|antelope_body"
	"Castoridae/beaver|beaver_body"
	"Bovidae/bison|bison_body"
	"Felidae/bobcat|bobcat_body"
	"Felidae/cat|cat_body"
	"Felidae/cheetah|cheetah_body"
	"Bovidae/cow|cow_body"
	"Cervidae/deer|deer_body"
	"Canidae/dog|dog_body"
	"Elephantidae/elephant|elephant_body"
	"Canidae/fox|fox_body"
	"Giraffidae/giraffe|giraffe_body"
	"Hominidae/gorilla|gorilla_body"
	"Cricetidae/hamster|hamster_body"
	"Hippopotamidae/hippo|hippo_body"
	"Equidae/horse|horse_body"
	"Equidae/zebra|zebra_body"
	"Felidae/leopard|leopard_body"
	"Felidae/lion|lion_body"
	"Mustelidae/otter|otter_body"
	"Ursidae/panda|panda_body"
	"Felidae/panther|panther_body"
	"Suidae/pig|pig_body"
	"Ursidae/polar bear|polar_bear_body"
	"Leporidae/rabbit|rabbit_body"
	"Procyonidae/raccoon|raccoon_body"
	"Muridae/rat|rat_body"
	"Rhinocerotidae/rhino|rhino_body"
	"Bovidae/sheep|sheep_body"
	"Mephitidae/skunk|skunk_body"
	"Cercopithecidae/spider monkey|spider_monkey_body"
	"Sciuridae/squirrel|squirrel_body"
	"Mustelidae/weasel|weasel_body"
	"Canidae/wolf|wolf_body"
)

for mapping in "${MAPPINGS[@]}"; do
	IFS='|' read -r src rel_link <<< "$mapping"
	src_path="$BASE_DIR/$src"
	if [[ -e "$src_path" ]]; then
		ln -sf "$src_path" "$rel_link"
		echo "Linked $rel_link -> $src_path"
	else
		echo "Warning: $src_path does not exist, skipping $rel_link"
	fi
done
