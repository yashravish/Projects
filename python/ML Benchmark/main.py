import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import requests
import zipfile
import shutil
import time
from joblib import Parallel, delayed
# Yash Ravish, yr288
DATA_URL = "http://rl.cs.rutgers.edu/fall2019/data.zip"
DATA_DIR = "data"

def download_and_extract_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    zip_path = os.path.join(DATA_DIR, "data.zip")
    if not os.path.exists(zip_path):
        print("Downloading dataset...")
        response = requests.get(DATA_URL, stream=True)
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(response.raw, f)
        print("Download complete.")
    
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("Extraction complete.")

def load_images_and_labels():
    digit_dir = os.path.join(DATA_DIR, "digitdata")
    face_dir = os.path.join(DATA_DIR, "facedata")

    def load_digit_data():
        try:
            with open(os.path.join(digit_dir, "trainingimages"), 'r') as f:
                lines = f.readlines()
                num_images = len(lines) // 28
                images = np.zeros((num_images, 28*28))
                for i in range(num_images):
                    image_lines = lines[i*28:(i+1)*28]
                    image_data = [1 if c != ' ' else 0 for line in image_lines for c in line[:28]]
                    images[i] = np.array(image_data)
            
            with open(os.path.join(digit_dir, "traininglabels"), 'r') as f:
                labels = np.array([int(label.strip()) for label in f.readlines()])
            
            return images, labels
            
        except Exception as e:
            print(f"Error loading digit data: {str(e)}")
            raise

    def load_face_data():
        try:
            with open(os.path.join(face_dir, "facedatatrain"), 'r') as f:
                lines = f.readlines()
                num_images = len(lines) // 70
                images = np.zeros((num_images, 70*60))
                for i in range(num_images):
                    image_lines = lines[i*70:(i+1)*70]
                    image_data = [1 if c != ' ' else 0 for line in image_lines for c in line[:60]]
                    images[i] = np.array(image_data)
            
            with open(os.path.join(face_dir, "facedatatrainlabels"), 'r') as f:
                labels = np.array([int(label.strip()) for label in f.readlines()])
            
            return images, labels
            
        except Exception as e:
            print(f"Error loading face data: {str(e)}")
            raise

    try:
        digit_images, digit_labels = load_digit_data()
        face_images, face_labels = load_face_data()
        return digit_images, digit_labels, face_images, face_labels
    
    except Exception as e:
        print(f"Error in load_images_and_labels: {str(e)}")
        raise

def extract_features(image, image_type='digit'):
    """Extract enhanced features from images."""
    features = []
    
    if image_type == 'digit':
        image = image.reshape(28, 28)
        
        features.extend(image.flatten())
        
        for i in range(0, 28, 4):
            for j in range(0, 28, 4):
                region = image[i:i+4, j:j+4]
                features.append(np.mean(region))  
                features.append(np.sum(region > 0))  
        
        h_proj = np.sum(image, axis=0) / 28
        v_proj = np.sum(image, axis=1) / 28
        features.extend(h_proj)
        features.extend(v_proj)
        
    else:  
        image = image.reshape(70, 60)
        
        downsampled = image[::2, ::2]
        features.extend(downsampled.flatten())
        
        for i in range(0, 70, 14):
            for j in range(0, 60, 12):
                region = image[i:i+14, j:j+12]
                features.append(np.mean(region))
                features.append(np.sum(region > 0) / (14 * 12))
        
        left_half = image[:, :30]
        right_half = np.fliplr(image[:, 30:])
        symmetry_score = np.mean(np.abs(left_half - right_half))
        features.append(symmetry_score)
    
    return np.array(features)

def preprocess_data(X, y, image_type='digit'):
    """Preprocess data and extract features."""
    X_features = np.array([extract_features(img, image_type) for img in X])
    
    mean = np.mean(X_features, axis=0)
    std = np.std(X_features, axis=0) + 1e-8
    X_normalized = (X_features - mean) / std
    
    return X_normalized, y

class NaiveBayes:
    def __init__(self):
        self.prior = None
        self.likelihood = None
        self.classes = None
        self.epsilon = 1e-10  

    def train(self, X_train, y_train):
        self.classes = np.unique(y_train)
        n_samples, n_features = X_train.shape
        
        class_counts = {c: np.sum(y_train == c) for c in self.classes}
        total_samples = len(y_train)
        self.prior = {c: (count + 1) / (total_samples + len(self.classes)) 
                     for c, count in class_counts.items()}
        
        self.likelihood = {}
        for c in self.classes:
            X_c = X_train[y_train == c]
            feature_counts = np.sum(X_c, axis=0)
            self.likelihood[c] = (feature_counts + 2) / (len(X_c) + 4)

    def predict(self, X_test):
        predictions = np.zeros(len(X_test))
        for i, x in enumerate(X_test):
            log_probs = {}
            for c in self.classes:
                likeli = self.likelihood[c]
                not_likeli = 1 - likeli
                
                likeli = np.clip(likeli, self.epsilon, 1-self.epsilon)
                not_likeli = np.clip(not_likeli, self.epsilon, 1-self.epsilon)
                
                log_probs[c] = (np.log(self.prior[c]) + 
                               np.sum(np.log(likeli) * x + 
                                     np.log(not_likeli) * (1 - x)))
            
            predictions[i] = max(log_probs, key=log_probs.get)
        return predictions

class Perceptron:
    def __init__(self, n_classes=10, learning_rate=0.01, epochs=500, batch_size=32):
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = None
        self.biases = None
        
    def _initialize_weights(self, n_features):
        self.weights = np.random.normal(0, 0.01, (self.n_classes, n_features))
        self.biases = np.zeros(self.n_classes)
    
    def _to_one_hot(self, y):
        one_hot = np.zeros((len(y), self.n_classes))
        for i, label in enumerate(y):
            one_hot[i, label] = 1
        return one_hot

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
    def train(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self._initialize_weights(n_features)
        
        y_one_hot = self._to_one_hot(y_train)
        best_weights = None
        best_biases = None
        best_accuracy = 0
        
        initial_lr = self.learning_rate
        
        for epoch in range(self.epochs):
            self.learning_rate = initial_lr / (1 + epoch * 0.01)
            
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_one_hot[indices]
            
            total_loss = 0
            
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                logits = np.dot(X_batch, self.weights.T) + self.biases
                probs = self._softmax(logits)
                
                error = y_batch - probs
                
                self.weights += self.learning_rate * np.dot(error.T, X_batch)
                self.biases += self.learning_rate * np.sum(error, axis=0)
                
                batch_loss = -np.mean(np.sum(y_batch * np.log(probs + 1e-15), axis=1))
                total_loss += batch_loss
            
            if epoch % 10 == 0:
                predictions = self.predict(X_train)
                accuracy = np.mean(predictions == y_train)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_weights = self.weights.copy()
                    best_biases = self.biases.copy()
                
                if accuracy > 0.95:
                    break
                
                if epoch > 0 and total_loss > prev_loss:
                    self.learning_rate *= 0.95  
                
            prev_loss = total_loss
        
        if best_weights is not None:
            self.weights = best_weights
            self.biases = best_biases
    
    def predict(self, X_test):
        logits = np.dot(X_test, self.weights.T) + self.biases
        return np.argmax(logits, axis=1)

    def predict_proba(self, X_test):
        logits = np.dot(X_test, self.weights.T) + self.biases
        return self._softmax(logits)

def split_data(images, labels, image_type='digit'):
    images = images / 255.0
    
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    X_train_processed, y_train = preprocess_data(X_train, y_train, image_type)
    X_test_processed, y_test = preprocess_data(X_test, y_test, image_type)
    
    return X_train_processed, X_test_processed, y_train, y_test

def prepare_random_subsets(X_train, y_train, steps=10, iterations=5):
    subsets = []
    total_size = len(X_train)
    for i in range(1, steps + 1):
        subset_size = total_size * i // steps
        subset = []
        for _ in range(iterations):
            indices = np.random.choice(total_size, subset_size, replace=False)
            subset.append((X_train[indices], y_train[indices]))
        subsets.append(subset)
    return subsets

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    start_time = time.time()
    model.train(X_train, y_train)
    training_time = time.time() - start_time
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test) * 100  
    return training_time, accuracy

def compare_algorithms(digit_subsets, digit_X_test, digit_y_test, face_subsets, face_X_test, face_y_test):
    results = {"NaiveBayes": [], "Perceptron": []}

    for subsets, X_test, y_test, label in [
        (digit_subsets, digit_X_test, digit_y_test, "Digit"),
        (face_subsets, face_X_test, face_y_test, "Face")
    ]:
        for model_name, Model in zip(["NaiveBayes", "Perceptron"], [NaiveBayes, Perceptron]):
            print(f"\nEvaluating {model_name} for {label} classification...")
            subset_results = []
            for subset in subsets:
                batch_results = Parallel(n_jobs=-1)(
                    delayed(train_and_evaluate)(Model(), X_train, y_train, X_test, y_test)
                    for X_train, y_train in subset
                )
                times, accuracies = zip(*batch_results)
                subset_results.append({
                    'times': np.mean(times),
                    'mean_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies)
                })

            print(f"Results for {model_name} ({label} Classification):")
            for i, r in enumerate(subset_results):
                print(f"  Training data: {(i+1)*10}%")
                print(f"    Mean accuracy: {r['mean_accuracy']:.2f}%")
                print(f"    Std deviation: {r['std_accuracy']:.2f}%")
                print(f"    Training time: {r['times']:.4f}s")

            results[model_name].append({
                "label": label,
                "training_times": [r['times'] for r in subset_results],
                "mean_accuracies": [r['mean_accuracy'] for r in subset_results],
                "std_accuracies": [r['std_accuracy'] for r in subset_results]
            })

    return results

def plot_learning_curves(results):
    for model_name, data in results.items():
        for entry in data:
            label = entry["label"]
            plt.figure(figsize=(10, 6))
            
            plt.subplot(1, 2, 1)
            plt.errorbar(range(10, 110, 10), entry["mean_accuracies"], 
                        yerr=entry["std_accuracies"], fmt='-o', color='blue', label='Accuracy')
            plt.title(f"{model_name} - {label} Classification Accuracy")
            plt.xlabel("Training Data (%)")
            plt.ylabel("Accuracy (%)")
            plt.grid(True)
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(range(10, 110, 10), entry["training_times"], 
                    color='red', marker='o', label='Training Time')
            plt.title(f"{model_name} - {label} Training Time")
            plt.xlabel("Training Data (%)")
            plt.ylabel("Time (seconds)")
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.show()

def main():
    print("Downloading and extracting data...")
    download_and_extract_data()
    
    print("Loading images and labels...")
    digit_images, digit_labels, face_images, face_labels = load_images_and_labels()
    
    print("\nPreprocessing digit data...")
    digit_X_train, digit_X_test, digit_y_train, digit_y_test = split_data(
        digit_images, digit_labels, image_type='digit'
    )
    
    print("Preprocessing face data...")
    face_X_train, face_X_test, face_y_train, face_y_test = split_data(
        face_images, face_labels, image_type='face'
    )
    
    print("\nPreparing training subsets...")
    digit_subsets = prepare_random_subsets(digit_X_train, digit_y_train)
    face_subsets = prepare_random_subsets(face_X_train, face_y_train)
    
    print("\nComparing algorithms...")
    results = compare_algorithms(
        digit_subsets, digit_X_test, digit_y_test,
        face_subsets, face_X_test, face_y_test
    )
    
    print("\nPlotting learning curves...")
    plot_learning_curves(results)
    
    print("\nExperiment complete!")

if __name__ == "__main__":
    main()