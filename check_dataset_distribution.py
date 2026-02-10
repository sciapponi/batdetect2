"""Check species distribution in training and validation datasets."""
import sys
from collections import Counter
from pathlib import Path

from soundevent import data

from batdetect2.data import load_dataset_from_config

def count_species(clip_annotations):
    """Count occurrences of each species in the dataset."""
    species_counter = Counter()
    clip_counter = Counter()
    
    for clip_annotation in clip_annotations:
        clip_species = set()
        for sound_event_annotation in clip_annotation.sound_events:
            # Look through all tags to find species information
            for tag in sound_event_annotation.tags:
                # Tags have term.label and value attributes
                term_label = tag.term.label if hasattr(tag, 'term') and hasattr(tag.term, 'label') else None
                
                # Check for Class tag (which contains species in batdetect2 datasets)
                if term_label == 'Class' and tag.value != 'Bat':  # Skip generic 'Bat' class
                    species = tag.value
                    species_counter[species] += 1
                    clip_species.add(species)
        
        # Count clips containing each species
        for species in clip_species:
            clip_counter[species] += 1
    
    return species_counter, clip_counter


def print_distribution(name, species_counter, clip_counter, total_clips):
    """Print formatted species distribution."""
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")
    print(f"Total clips: {total_clips}")
    print(f"Total sound events: {sum(species_counter.values())}")
    print(f"\n{'Species':<20} {'Events':<10} {'Clips':<10} {'Events/Clip':<12}")
    print("-" * 80)
    
    for species, count in sorted(species_counter.items(), key=lambda x: x[1], reverse=True):
        clips = clip_counter[species]
        events_per_clip = count / clips if clips > 0 else 0
        print(f"{species:<20} {count:<10} {clips:<10} {events_per_clip:<12.2f}")
    
    print("-" * 80)
    print(f"{'TOTAL':<20} {sum(species_counter.values()):<10} "
          f"{sum(clip_counter.values()):<10}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_dataset_distribution.py <train_dataset.yaml> [val_dataset.yaml]")
        print("\nExample:")
        print("  python check_dataset_distribution.py example_data/uk_diff_train.yaml example_data/uk_diff_val.yaml")
        sys.exit(1)
    
    train_path = Path(sys.argv[1])
    val_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    print("Loading datasets...")
    print(f"Train: {train_path}")
    
    # Load training dataset
    train_dataset = load_dataset_from_config(train_path)
    train_species, train_clips = count_species(train_dataset)
    
    print_distribution(
        "TRAINING DATASET",
        train_species,
        train_clips,
        len(train_dataset)
    )
    
    # Load validation dataset if provided
    if val_path:
        print(f"\nVal:   {val_path}")
        val_dataset = load_dataset_from_config(val_path)
        val_species, val_clips = count_species(val_dataset)
        
        print_distribution(
            "VALIDATION DATASET",
            val_species,
            val_clips,
            len(val_dataset)
        )
        
        # Compare distributions
        print(f"\n{'='*80}")
        print("COMPARISON")
        print(f"{'='*80}")
        print(f"{'Species':<20} {'Train Events':<15} {'Val Events':<15} {'Train/Val Ratio':<15}")
        print("-" * 80)
        
        all_species = set(train_species.keys()) | set(val_species.keys())
        for species in sorted(all_species):
            train_count = train_species.get(species, 0)
            val_count = val_species.get(species, 0)
            ratio = train_count / val_count if val_count > 0 else float('inf')
            ratio_str = f"{ratio:.2f}" if ratio != float('inf') else "∞"
            
            # Highlight species missing from validation
            marker = " ⚠️" if val_count == 0 else ""
            print(f"{species:<20} {train_count:<15} {val_count:<15} {ratio_str:<15}{marker}")
        
        print("-" * 80)
        
        # Summary of missing species
        missing_in_val = set(train_species.keys()) - set(val_species.keys())
        missing_in_train = set(val_species.keys()) - set(train_species.keys())
        
        if missing_in_val:
            print(f"\n⚠️  Species in training but NOT in validation: {', '.join(sorted(missing_in_val))}")
        if missing_in_train:
            print(f"\n⚠️  Species in validation but NOT in training: {', '.join(sorted(missing_in_train))}")


if __name__ == "__main__":
    main()
