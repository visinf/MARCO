#!/bin/bash

# Prepare DeepFashion2 apparel categories for the MP-100 benchmark.

src="train/image"

classes=(
  short_sleeved_outwear
  short_sleeved_shirt
  skirt
  short_sleeved_dress
  vest_dress
  long_sleeved_dress
  long_sleeved_outwear
  long_sleeved_shirt
  sling
  sling_dress
  trousers
  vest
)

for c in "${classes[@]}"; do
  mkdir -p "$c"
  cp -al "$src"/. "$c"/
done
