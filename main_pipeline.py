import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------- CONFIGURATION ----------------
BASE_PATH = 'archive_2/WESAD' 
# ------------------------------------------------

def load_subject_data(subject_id):
    """WESAD .pkl ফাইল লোড করার ফাংশন"""
    folder_name = f"S{subject_id}"
    filename = f"S{subject_id}.pkl"
    filepath = os.path.join(BASE_PATH, folder_name, filename)
    
    if not os.path.exists(filepath):
        return None
        
    try:
        with open(filepath, 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_features(signals, labels):
    """Chest Device থেকে EDA এবং ECG ফিচার এক্সট্রাক্ট করা"""
    features = []
    target_labels = []
    
    # Chest Device সিগন্যাল নেওয়া
    chest_signals = signals['chest']
    eda_signal = chest_signals['EDA'].flatten() # 1D Array এ কনভার্ট
    ecg_signal = chest_signals['ECG'].flatten()
    
    # WESAD স্যাম্পল রেট ৭০Hz, ৫ সেকেন্ড উইন্ডো = ৩৫০ স্যাম্পল
    window_size = 350 
    
    for i in range(0, len(eda_signal) - window_size, window_size):
        eda_win = eda_signal[i:i+window_size]
        ecg_win = ecg_signal[i:i+window_size]
        
        mid_idx = i + window_size // 2
        if mid_idx < len(labels):
            label = int(labels[mid_idx])
            
            # শুধুমাত্র Neutral (0), Stress (1), Amusement (2) নেওয়া
            if label in [0, 1, 2]:
                # --- EDA Features ---
                mean_eda = np.mean(eda_win)
                std_eda = np.std(eda_win)
                max_eda = np.max(eda_win)
                min_eda = np.min(eda_win)
                
                # --- ECG Features ---
                mean_ecg = np.mean(ecg_win)
                std_ecg = np.std(ecg_win)
                
                # Feature Vector
                features.append([mean_eda, std_eda, max_eda, min_eda, mean_ecg, std_ecg])
                target_labels.append(label)
            
    return np.array(features), np.array(target_labels)

# ---------------- MAIN EXECUTION ----------------
print("Starting Emotion Recognition Pipeline...")
print(f"Looking for data in: {BASE_PATH}")
all_features = []
all_labels = []

# Subject 2 to 17 লোড করা
for sub_id in range(2, 18):
    data = load_subject_data(sub_id)
    if data:
        try:
            signals = data['signal']
            labels = data['label']
            
            feats, labs = extract_features(signals, labels)
            if len(feats) > 0:
                all_features.append(feats)
                all_labels.append(labs)
                print(f"Subject {sub_id} processed. Samples: {len(feats)}")
        except KeyError as e:
            print(f"Key Error in Subject {sub_id}: {e}")
        except Exception as e:
            print(f"Unexpected error in Subject {sub_id}: {e}")
    else:
        print(f"Subject {sub_id} data not found.")

if all_features:
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    print(f"\nTotal Samples Generated: {X.shape[0]}")
    print(f"Unique Labels: {np.unique(y)}")
    print("Label Map: 0=Neutral, 1=Stress, 2=Amusement")
    
    # Train-Test Split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- Model 1: SVM ---
    print("\nTraining SVM...")
    svm_model = SVC(kernel='rbf', C=10, gamma='scale')
    svm_model.fit(X_train_scaled, y_train)
    y_pred_svm = svm_model.predict(X_test_scaled)
    
    # --- Model 2: Random Forest ---
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    y_pred_rf = rf_model.predict(X_test_scaled)
    
    # --- Results ---
    acc_svm = accuracy_score(y_test, y_pred_svm)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    
    print(f"\n{'='*30}")
    print(f"FINAL RESULTS")
    print(f"{'='*30}")
    print(f"SVM Accuracy:          {acc_svm:.2f}")
    print(f"Random Forest Accuracy: {acc_rf:.2f}")
    
    print("\nSVM Classification Report:")
    print(classification_report(y_test, y_pred_svm, target_names=['Neutral', 'Stress', 'Amusement']))
    
    # --- Save Confusion Matrix Image ---
    cm = confusion_matrix(y_test, y_pred_svm)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Neutral', 'Stress', 'Amusement'], 
                yticklabels=['Neutral', 'Stress', 'Amusement'])
    plt.title('Confusion Matrix (SVM) - WESAD Dataset')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.tight_layout()
    plt.savefig('result_confusion_matrix.png')
    print("\nConfusion Matrix saved as 'result_confusion_matrix.png'")

else:
    print("\nError: No data was loaded. Please check the BASE_PATH and file structure.")
    import joblib

# ... (তোমার আগের সব কোড ঠিক থাকবে) ...

# মডেল ট্রেইনিং শেষ হওয়ার পর এই লাইনগুলো যোগ করো:
joblib.dump(rf_model, 'emotion_rf_model.pkl')  # Random Forest মডেল সেভ
joblib.dump(scaler, 'scaler.pkl')              # স্কেলার সেভ
print("✅ Model and Scaler saved successfully!")