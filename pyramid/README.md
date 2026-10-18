# pyramid

Out-of-core brick pyramid builder. Reads the source volume slab by slab (raw/NRRD memmap) and writes compressed, sharded bricks per level: in-plane 2x2x1 downsampling until the in-plane spacing reaches the section spacing, then 2x2x2. Memory stays bounded by the slab size.
