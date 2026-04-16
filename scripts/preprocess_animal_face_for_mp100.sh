#!/bin/bash

# Prepare AnimalWeb animal-face images for the MP-100 benchmark.
# Moves the selected category images into per-category folders.

categories=(
  alpaca_face
  californiansealion_face
  chipmunk_face
  ferret_face
  gibbons_face
  guanaco_face
  proboscismonkey_face
  arcticwolf_face
  camel_face
  commonwarthog_face
  gentoopenguin_face
  greyseal_face
  klipspringer_face
  fennecfox_face
  blackbuck_face
  capebuffalo_face
  dassie_face
  gerbil_face
  grizzlybear_face
  olivebaboon_face
  quokka_face
  bonobo_face
  capybara_face
  fallowdeer_face
  onager_face
  pademelon_face
)

src="OpenDataLab___AnimalWeb/animal_dataset_v1_clean_check"
for cat in "${categories[@]}"; do
  base="${cat%_face}"
  mkdir -p "$cat"
  mv "$src"/${base}_*.jpg "$cat"/ 2>/dev/null || true
done