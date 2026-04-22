"""Demonstrate proper train/validation/test split strategy."""

from cell_tracking.splits import train_validation_split


def main():
    """Show the recommended frame split for the project."""
    
    total_frames = 92  # Fluo-N2DH-GOWT1 track 01 has 92 frames
    
    # Example splits (60% train, 20% val, 20% test)
    train_frames, val_frames, test_frames = train_validation_split(
        total_frames=total_frames,
        train_fraction=0.6,
        val_fraction=0.2
    )
    
    print("Recommended Frame Split (60/20/20):")
    print(f"  Training frames ({len(train_frames)}):   {train_frames[0]}-{train_frames[-1]}")
    print(f"  Validation frames ({len(val_frames)}):   {val_frames[0]}-{val_frames[-1]}")
    print(f"  Test frames ({len(test_frames)}):      {test_frames[0]}-{test_frames[-1]}")
    print()
    
    # Show as frame spec for use with --frames argument
    train_spec = f"0-{train_frames[-1]}"
    val_spec = f"{val_frames[0]}-{val_frames[-1]}"
    test_spec = f"{test_frames[0]}-{test_frames[-1]}"
    
    print("Frame specs for scripts:")
    print(f"  Training:   --frames '{train_spec}'")
    print(f"  Validation: --frames '{val_spec}'")
    print(f"  Testing:    --frames '{test_spec}'")
    print()
    
    print("Example training command:")
    print(f"  python scripts/train_model.py Fluo-N2DH-GOWT1 --frames '{train_spec}' --model svm")


if __name__ == "__main__":
    main()
