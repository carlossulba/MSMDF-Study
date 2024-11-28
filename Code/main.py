# main.py

from Classes.FingerprintExtractor import FingerprintExtractor
from Classes.ClassifierTrainer import ClassifierTrainer
from Classes.ClassifierEvaluator import ClassifierEvaluator
from Classes.CountermeasureApplier import CountermeasureApplier

def main():
    # Step 1: Extract fingerprints
    data_path = "../Data/raw data/all data"
    
    data_streams = [
        # Accelerometer
        "acc_ist", "acc_az", "acc_incl",
        "acc_x", "acc_y", "acc_z", "acc_mag",
        # Accelerometer with gravity
        "acc_grav_ist", "acc_grav_az", "acc_grav_incl",
        "acc_grav_x", "acc_grav_y", "acc_grav_z", "acc_grav_mag",
        # Gyroscope
        "gyro_x", "gyro_y", "gyro_z", "gyro_mag"
    ]
    
    features = [
        # Time domain features
        "avg_dev", "kurtosis", "max", "mean", "min", "mode",
        "non_negative_count", "range", "rms", "skewness",
        "std_dev", "var", "zcr",
        
        # Frequency domain features
        "dc_offset", "irregularity_j", "irregularity_k", "low_energy_rate", "smoothness",
        "spec_attack_slope", "spec_attack_time", "spec_brightness", "spec_centroid", "spec_crest",
        "spec_entropy", "spec_flatness", "spec_flux", "spec_irregularity", "spec_kurtosis",
        "spec_rms", "spec_roll_off", "spec_roughness", "spec_skewness", "spec_spread",
        "spec_std_dev", "spec_var"
    ]
    
    extractor = FingerprintExtractor(data_path, data_streams, features)
    fingerprints = extractor.extract_fingerprint()

    """
    # Step 2: Train classifiers
    classifier_type = "SVM"  # or "RandomForest"
    trainer = ClassifierTrainer(fingerprints, classifier_type, C=1.0, kernel='linear')
    classifier = trainer.train_classifiers()

    # Step 3: Evaluate classifiers
    evaluator = ClassifierEvaluator({'main_classifier': classifier})
    # Assuming features and labels are the test dataset
    test_features, test_labels = trainer.get_features_and_labels()
    performance = evaluator.evaluate_classifier(test_features, test_labels)
    print("Evaluation Results:", performance)

    # Future Step: Apply countermeasures (currently a placeholder)
    countermeasure_applier = CountermeasureApplier()
    # Example: processed_data = countermeasure_applier.apply_countermeasure(raw_data)
    """

if __name__ == "__main__":
    main()
