# Import necessary libraries
import os
import pandas as pd
import numpy as np
from time import time
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,roc_curve, auc)
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting (saving figures to files)
import matplotlib.pyplot as plt
import matplotlib.colors
import xgboost as xgb
import seaborn as sns
import traceback # For detailed error reporting

# --- Global Variables ---
df: DataFrame # Holds the main dataset throughout preprocessing
df_pca_result: DataFrame # Holds the PCA-transformed data
df_lda_result: DataFrame # Holds the LDA-transformed data
base_dir: str = "" # Path to the dataset file
# Columns selected for specific missing value percentages based on project requirements
cols_5_percent = ['MajorAxisLength', 'EquivDiameter'] # Columns to add ~5% missing values
col_35_percent = 'Solidity' # Column to add ~35% missing values
le: LabelEncoder # LabelEncoder instance for the target variable ('Class')

def save_plot(filename: str):
    """
    Saves the current matplotlib plot to the 'outputs' folder.
    Creates the 'outputs' folder if it doesn't exist.
    Assumes a project structure where the script is in a subfolder (e.g., 'src')
    and 'outputs' is parallel to it. Handles cases where __file__ might not be defined.
    """
    try:
        # Determine the script's directory
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError: # Handle cases like running in an interactive environment
            script_dir = os.getcwd()
            print(f"Warning: __file__ not defined, using current working directory: {script_dir}")

        # Navigate up to the project directory and define the output folder path
        project_dir = os.path.dirname(script_dir) # Assumes script is one level down from project root
        output_folder = os.path.join(project_dir, "outputs")
        os.makedirs(output_folder, exist_ok=True) # Create the folder if it doesn't exist
        output_path = os.path.join(output_folder, filename)
        plt.savefig(output_path)
        print(f"Plot saved as '{output_path}'.")
    except Exception as e:
        print(f"Error saving plot '{filename}': {e}")
    finally:
        plt.close() # Close the plot figure to free memory

# --- Part 1: Data Preprocessing Functions ---

def load_dataset():
    """
    Loads the Dry Bean Dataset from the specified Excel file.
    Sets the global 'df' DataFrame and 'base_dir'.
    Prints basic dataset information (head, info).
    Handles FileNotFoundError. Corresponds to Project Step 1.
    """
    global df, base_dir
    try:
        # Determine the script's directory (handle interactive environments)
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = os.getcwd()
            print(f"Warning: __file__ not defined, using current working directory: {script_dir}")

        # Construct the path to the dataset file within the 'data' folder
        project_dir = os.path.dirname(script_dir)
        base_dir = os.path.join(project_dir, "data", "Dry_Bean_Dataset.xlsx")
        print(f"Dataset path: {base_dir}")

        # Check if the file exists before attempting to read
        if not os.path.exists(base_dir):
             print(f"Error: Path '{base_dir}' not found.")
             exit() # Exit if dataset is not found

        # Read the Excel file into the global DataFrame 'df'
        df = pd.read_excel(base_dir)

        # Display initial information about the loaded dataset
        print("Dataset loaded successfully.")
        print("First 5 rows:")
        print(df.head())
        print("\nDataset info:")
        df.info()
        print("-" * 30)
    except FileNotFoundError:
        print(f"Error: 'Dry_Bean_Dataset.xlsx' not found in '{os.path.dirname(base_dir)}'.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        exit()

def add_missing_values():
    """
    Adds missing values (NaN) to the dataset as per Project Step 2.
    - Adds ~5% missing values to columns specified in 'cols_5_percent'.
    - Adds ~35% missing values to the column specified in 'col_35_percent'.
    Uses fixed random seeds for reproducibility.
    Prints the count of missing values added and observes the result using isnull().sum() (Step 2a).
    """
    global df
    if df is None: print("Error: DataFrame 'df' is not loaded!"); return
    print("\n--- Step 2: Adding Missing Values ---")
    np.random.seed(42) # Seed for the 5% missing values
    random_state_35 = np.random.RandomState(1) # Separate RandomState for 35%

    # Add ~5% missing values to specified columns
    for col in cols_5_percent:
        if col in df.columns:
            missing_indices = df.sample(frac=0.05, random_state=np.random.RandomState(42)).index
            df.loc[missing_indices, col] = np.nan
            print(f"~5% ({len(missing_indices)}) missing values added to column '{col}'.")
        else:
            print(f"Warning: Column '{col}' not found in DataFrame, could not add missing values.")

    # Add ~35% missing values to the specified column
    if col_35_percent in df.columns:
        missing_indices_35 = df.sample(frac=0.35, random_state=random_state_35).index
        df.loc[missing_indices_35, col_35_percent] = np.nan
        print(f"~35% ({len(missing_indices_35)}) missing values added to column '{col_35_percent}'.")
    else:
        print(f"Warning: Column '{col_35_percent}' not found in DataFrame, could not add missing values.")

    # Observe missing values after addition (Project Step 2a)
    print("\nStep 2a: Observing Missing Values (isnull().sum())")
    print(df.isnull().sum())

def fill_missing_values():
    """
    Handles the missing values added in the previous step, following Project Step 2b/c.
    - Step 2b: Fills columns with ~5% missing values using the median.
    - Step 2c: Decides how to handle the column with ~35% missing values:
        - If missing percentage > 30%, drops the column.
        - Otherwise, fills with the median.
    Prints the actions taken and the final missing value counts.
    """
    global df
    if df is None: print("Error: DataFrame 'df' is not loaded!"); return
    print("\n--- Step 2b/c: Filling/Dropping Missing Values ---")

    # Step 2b: Fill columns with ~5% missing values using median
    print("\nStep 2b: Filling Columns with 5% Missing Values using Median")
    for col in cols_5_percent:
         if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val) # Fill NaN with median
            print(f"Missing values in column '{col}' filled with median ({median_val:.2f}).")
         elif col not in df.columns:
             print(f"Warning: Column '{col}' not found for filling.")

    print("\nMissing value status after 5% fill:")
    print(df.isnull().sum())

    # Step 2c: Handle the column with ~35% missing values
    print(f"\nStep 2c: Decision for Column with 35% Missing Values ('{col_35_percent}')")
    if col_35_percent in df.columns:
        missing_percentage = df[col_35_percent].isnull().mean() * 100
        print(f"Missing value percentage in '{col_35_percent}': {missing_percentage:.2f}%")

        threshold = 30.0 # Threshold for dropping the column
        if missing_percentage > threshold:
            print(f"Missing value rate in '{col_35_percent}' ({missing_percentage:.2f}%) > {threshold}%, dropping column.")
            df.drop(col_35_percent, axis=1, inplace=True) # Drop column if too many missing
        elif df[col_35_percent].isnull().any():
             median_val_35 = df[col_35_percent].median()
             df[col_35_percent].fillna(median_val_35, inplace=True) # Fill with median otherwise
             print(f"Missing value rate in '{col_35_percent}' ({missing_percentage:.2f}%) <= {threshold}%, filled with median ({median_val_35:.2f}).")
        else:
            print(f"No missing values found in '{col_35_percent}' to fill.")

    else:
        print(f"Warning: Column '{col_35_percent}' not found for drop/fill decision.")

    # Final check on missing values
    print(f"\nMissing value status after handling '{col_35_percent}':")
    print(df.isnull().sum())
    print("\nCurrent dataset shape:", df.shape)

def outlier_detection():
    """
    Detects and handles outliers in numerical columns using the IQR method (Project Step 3).
    - Calculates Q1, Q3, and IQR for each numerical column (excluding target 'Class').
    - Identifies values outside the [Q1 - 1.5*IQR, Q3 + 1.5*IQR] range as outliers.
    - Replaces detected outliers with the boundary values (clipping).
    Prints the number of outliers detected and handled per column.
    Returns the list of numerical columns processed.
    """
    global df
    if df is None: print("Error: DataFrame 'df' is not loaded!"); return None
    print("\n--- Step 3: Outlier Detection and Handling (IQR Method) ---")

    # Select numerical columns, excluding the target variable 'Class' if present
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    target_col = 'Class'
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)

    if not numerical_cols:
        print("No numerical columns found for outlier detection.")
        return []

    print(f"Numerical columns to check for outliers: {numerical_cols}")
    outlier_cols_found = [] # Keep track of columns where outliers were handled

    # Iterate through numerical columns to detect and handle outliers
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Skip if IQR is 0 (e.g., constant column)
        if IQR == 0:
            print(f"IQR is 0 for column '{col}', skipping outlier check.")
            continue

        # Define outlier boundaries
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Find outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        n_outliers = outliers.shape[0]

        # Handle outliers if found
        if n_outliers > 0:
            print(f"{n_outliers} outliers detected in column '{col}'.")
            # Replace outliers with boundary values (clipping)
            df[col] = np.clip(df[col], lower_bound, upper_bound)
            print(f"--> Outliers in column '{col}' replaced with boundary values [{lower_bound:.2f}, {upper_bound:.2f}] (clipping).")
            outlier_cols_found.append(col)

    if not outlier_cols_found:
        print("No outliers were detected in the numerical columns.")
    print("\nOutlier handling completed.")
    return numerical_cols # Return the list of columns checked

def feature_scaling(numerical_cols_to_scale: list):
    """
    Feature scaling is important to standardize the range of numerical features.
    This prevents features with larger ranges from disproportionately influencing
    distance-based algorithms (like PCA, LDA, LR, KNN) or algorithms using gradient descent.
    It often leads to better model performance and faster convergence.
    Applies feature scaling using StandardScaler to the specified numerical columns (Project Step 4).
    Prints an explanation of why scaling is important.
    Shows the first 5 rows of scaled data and checks mean/std dev of one column.
    """
    global df
    if df is None: print("Error: DataFrame 'df' is not loaded!"); return
    if not numerical_cols_to_scale:
        print("List of numerical columns to scale is empty or not provided.")
        return
    # Ensure columns exist in the DataFrame
    valid_cols_to_scale = [col for col in numerical_cols_to_scale if col in df.columns]
    if not valid_cols_to_scale:
        print("No valid numerical columns found in the DataFrame to scale.")
        return

    print("\n--- Step 4: Feature Scaling (StandardScaler) ---")
    print(f"The following numerical columns will be scaled using StandardScaler: {valid_cols_to_scale}")

    # Initialize and apply StandardScaler
    scaler = StandardScaler()
    df[valid_cols_to_scale] = scaler.fit_transform(df[valid_cols_to_scale])

    # Display results
    print("\nFirst 5 rows after scaling (numerical columns):")
    print(df[valid_cols_to_scale].head())

    # Verify scaling (optional check)
    if valid_cols_to_scale:
        print(f"\nMean of '{valid_cols_to_scale[0]}' after scaling: {df[valid_cols_to_scale[0]].mean():.4f}") # Should be close to 0
        print(f"Standard deviation of '{valid_cols_to_scale[0]}' after scaling: {df[valid_cols_to_scale[0]].std():.4f}") # Should be close to 1

def categorical_encoding():
    """
    Encodes categorical variables (Project Step 5).
    - Uses LabelEncoder for the target column 'Class' and stores the encoder in global 'le'.
    - Uses pandas get_dummies (One-Hot Encoding) for any other remaining 'object' type columns.
    Prints mapping for LabelEncoder and info after encoding.
    """
    global df, le
    if df is None: print("Error: DataFrame 'df' is not loaded!"); return
    print("\n--- Step 5: Categorical Encoding ---")
    target_col = 'Class'

    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in DataFrame.")
        return

    # Encode the target variable 'Class' using LabelEncoder
    if df[target_col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[target_col]):
        print(f"Target column '{target_col}' is categorical. Applying LabelEncoder.")
        le = LabelEncoder() # Initialize LabelEncoder
        df[target_col] = le.fit_transform(df[target_col]) # Fit and transform

        # Print the mapping from original class names to numerical labels
        print("\nLabelEncoder Class Mappings:")
        try:
            class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            print(class_mapping)
        except Exception as e:
            print(f"Could not retrieve class mapping: {e}")

        print(f"\nFirst 5 values of '{target_col}' after encoding:")
        print(df[target_col].head())
        print(f"Unique values in '{target_col}' after encoding: {np.sort(df[target_col].unique())}")
    else:
        # Handle case where target is already numerical (e.g., if script is rerun)
        print(f"Target column '{target_col}' is already numerical or LabelEncoder was previously applied.")
        # If it's numerical but 'le' wasn't created, create it now for consistency (needed for ROC labels later)
        if le is None and pd.api.types.is_numeric_dtype(df[target_col]):
             print("Creating LabelEncoder instance for existing numerical target variable...")
             le = LabelEncoder()
             unique_target_values = np.sort(df[target_col].unique())
             le.fit(unique_target_values) # Fit on the existing numerical values
             print("LabelEncoder Class Mappings (Existing Numerical Values):")
             try:
                 class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                 print(class_mapping)
             except Exception as e:
                 print(f"Could not retrieve class mapping: {e}")

    # Encode other potential categorical features using One-Hot Encoding
    categorical_features = df.select_dtypes(include='object').columns.tolist()
    if categorical_features:
        print(f"\nApplying One-Hot Encoding for other categorical features: {categorical_features}...")
        df = pd.get_dummies(df, columns=categorical_features, drop_first=True) # drop_first=True to avoid multicollinearity
        print("DataFrame head after One-Hot Encoding:")
        print(df.head())
        print("\nDataFrame info after One-Hot Encoding:")
        df.info()
    else:
        print("\nNo other categorical input features of type 'object' found.")

    print("\n--- Categorical Encoding Step Completed ---")

def part1():
    """
    Executes all data preprocessing steps defined in Part 1 of the project.
    Calls functions for loading, missing value handling, outlier detection,
    feature scaling, and categorical encoding.
    """
    print("\n" + "="*10 + " Part 1: Data Loading and Preprocessing " + "="*10)
    load_dataset() # Step 1
    if df is not None:
        add_missing_values() # Step 2 (add), 2a (observe)
        fill_missing_values() # Step 2b (fill 5%), 2c (handle 35%)
        numerical_cols_list = outlier_detection() # Step 3
        if numerical_cols_list is not None:
            feature_scaling(numerical_cols_list) # Step 4
        else:
            print("Skipping feature scaling as no numerical columns were identified or returned from outlier detection.")
        categorical_encoding() # Step 5
        print("\n--- Part 1 Completed ---")
        print("\nFinal DataFrame Info:")
        df.info()
        print("\nFinal DataFrame First 5 Rows:")
        print(df.head())
    else:
        print("Part 1 operations could not be performed because data loading failed.")

# --- Part 2: Feature Extraction Functions ---

def apply_pca(X_data: pd.DataFrame, y_data: pd.Series, target_col_name: str, random_state: int = 42):
    """
    Applies Principal Component Analysis (PCA) for dimensionality reduction (Project Step 6).
    - Determines the number of components where explained variance > average explained variance.
    - Fits PCA with the selected number of components.
    - Generates an explained variance plot.
    - Generates a 2D scatter plot of the first two principal components.
    - Returns the PCA-transformed data as a DataFrame, the PCA model, and the number of components.
    """
    print("\n--- Step 6: Applying PCA ---")
    if X_data is None or y_data is None or X_data.empty:
        print("Error: X or y data is missing or empty for PCA.")
        return None, None, None

    try:
        # Fit PCA with all components initially to analyze explained variance
        pca_full = PCA(random_state=random_state)
        pca_full.fit(X_data)
        explained_variance_ratio = pca_full.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        avg_explained_variance = np.mean(explained_variance_ratio)
        print(f"Average explained variance ratio per component: {avg_explained_variance:.4f}")

        # Select number of components based on the project requirement (variance > average)
        n_components = np.sum(explained_variance_ratio > avg_explained_variance)
        if n_components == 0: n_components = 1 # Ensure at least one component
        print(f"Selected number of components (variance > average): {n_components}")

        # Apply PCA with the selected number of components
        pca_model = PCA(n_components=n_components, random_state=random_state)
        X_pca = pca_model.fit_transform(X_data)
        print(f"Shape after PCA: {X_pca.shape}")

        # Create a DataFrame for the PCA results
        pca_columns = [f'PC{i+1}' for i in range(n_components)]
        df_pca = pd.DataFrame(data=X_pca, columns=pca_columns, index=X_data.index)
        df_pca[target_col_name] = y_data.values # Add the target variable back
        print("\nPCA transformed data first 5 rows:")
        print(df_pca.head())

        # Plot Explained Variance (Project Step 6 requirement)
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, align='center', label='Individual Explained Variance')
        plt.step(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, where='mid', label='Cumulative Explained Variance', color='red')
        plt.ylabel('Explained Variance Ratio')
        plt.xlabel('Number of Principal Components')
        plt.title('PCA Explained Variance Plot')
        plt.axhline(y=avg_explained_variance, color='orange', linestyle=':', label=f'Average Variance ({avg_explained_variance:.4f})') # Show average line
        plt.axvline(x=n_components, color='purple', linestyle='--', label=f'Selected Components ({n_components})') # Show selected components
        plt.xticks(range(1, len(explained_variance_ratio) + 1))
        plt.legend(loc='best')
        plt.grid(True, axis='y')
        plt.tight_layout()
        save_plot("pca_explained_variance.png")

        # Plot 2D Scatter Plot of first two components (Project Step 6 requirement)
        if n_components >= 2:
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=df_pca[pca_columns[0]], y=df_pca[pca_columns[1]], hue=y_data, palette='viridis', s=50, alpha=0.7)
            plt.title('PCA Dimensionality Reduction (First 2 Principal Components)')
            plt.xlabel(f'Principal Component 1 ({pca_columns[0]})')
            plt.ylabel(f'Principal Component 2 ({pca_columns[1]})')
            plt.legend(title=target_col_name)
            plt.grid(True)
            plt.tight_layout()
            save_plot("pca_scatter_2d.png")
        elif n_components == 1:
             print("Only 1 PCA component selected, 2D visualization not possible.")

        print("\nPCA application completed.")
        return df_pca, pca_model, n_components

    except Exception as e:
        print(f"An error occurred during PCA application: {e}")
        traceback.print_exc() # Print full traceback for debugging
        return None, None, None

def apply_lda(X_data: pd.DataFrame, y_data: pd.Series, target_col_name: str, n_components_lda: int = 3):
    """
    Applies Linear Discriminant Analysis (LDA) for dimensionality reduction (Project Step 7).
    - Uses a fixed number of components (n_components_lda=3 as specified), but adjusts if needed.
    - Fits LDA to the data.
    - Generates a 2D scatter plot of the first two linear discriminants.
    - Generates a 1D histogram if only 1 component is produced.
    - Generates a 3D scatter plot if 3 or more components are produced.
    - Returns the LDA-transformed data as a DataFrame and the LDA model.
    """
    print(f"\n--- Step 7: Applying LDA ---")
    if X_data is None or y_data is None or X_data.empty:
        print("Error: X or y data is missing or empty for LDA.")
        return None, None

    try:
        n_features = X_data.shape[1]
        unique_classes = np.unique(y_data)
        n_classes = len(unique_classes)
        print(f"Number of unique classes: {n_classes}")

        # LDA requires at least 2 classes
        if n_classes <= 1:
            print("Error: LDA requires at least 2 classes.")
            return None, None

        # The maximum number of components for LDA is min(n_features, n_classes - 1)
        max_lda_components = min(n_features, n_classes - 1)

        # Adjust the requested number of components if necessary
        if n_components_lda > max_lda_components:
            n_components_final = max_lda_components
            print(f"Warning: Requested LDA components ({n_components_lda}) > max possible ({max_lda_components}). Using {n_components_final}.")
        elif n_components_lda <= 0:
             n_components_final = max_lda_components # Use max if invalid number requested
             print(f"Warning: Invalid number of LDA components requested ({n_components_lda}). Using max possible ({n_components_final}).")
        else:
            n_components_final = n_components_lda # Use the requested number (3)

        print(f"Number of LDA components to use: {n_components_final}")

        # Check if a valid number of components could be determined
        if n_components_final <= 0:
             print("Error: Cannot determine a valid number of LDA components.")
             return None, None

        # Initialize and apply LDA
        lda_model = LinearDiscriminantAnalysis(n_components=n_components_final)
        X_lda = lda_model.fit_transform(X_data, y_data) # Note: LDA uses y during fit
        print(f"Shape after LDA: {X_lda.shape}")

        # Create a DataFrame for the LDA results
        lda_columns = [f'LD{i+1}' for i in range(n_components_final)]
        df_lda = pd.DataFrame(data=X_lda, columns=lda_columns, index=X_data.index)
        df_lda[target_col_name] = y_data.values # Add the target variable back
        print("\nLDA transformed data first 5 rows:")
        print(df_lda.head())

        # Plot 2D Scatter Plot of first two components (Project Step 7 requirement)
        if n_components_final >= 2:
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=df_lda[lda_columns[0]], y=df_lda[lda_columns[1]], hue=y_data, palette='viridis', s=50, alpha=0.7)
            plt.title('LDA Dimensionality Reduction (First 2 Linear Discriminants)')
            plt.xlabel(f'Linear Discriminant 1 ({lda_columns[0]})')
            plt.ylabel(f'Linear Discriminant 2 ({lda_columns[1]})')
            plt.legend(title=target_col_name)
            plt.grid(True)
            plt.tight_layout()
            save_plot("lda_scatter_2d.png")

        # Plot 1D Histogram if only one component is generated
        elif n_components_final == 1:
             print("Only 1 LDA component generated, saving 1D histogram.")
             plt.figure(figsize=(10, 6))
             sns.histplot(data=df_lda, x=lda_columns[0], hue=target_col_name, kde=True, palette='viridis')
             plt.title('LDA Dimensionality Reduction (1 Linear Discriminant)')
             plt.xlabel(f'Linear Discriminant 1 ({lda_columns[0]})')
             plt.tight_layout()
             save_plot("lda_hist_1d.png")

        # Optional: Plot 3D Scatter Plot if 3 components are generated
        if n_components_final >= 3:
            print("\nLDA 3D Visualization (First 3 Components):")
            fig = plt.figure(figsize=(12, 9))
            try:
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(df_lda[lda_columns[0]], df_lda[lda_columns[1]], df_lda[lda_columns[2]],
                                     c=y_data, cmap='viridis', s=40, alpha=0.6)
                ax.set_title('LDA Dimensionality Reduction (First 3 Linear Discriminants)')
                ax.set_xlabel(f'Linear Discriminant 1 ({lda_columns[0]})')
                ax.set_ylabel(f'Linear Discriminant 2 ({lda_columns[1]})')
                ax.set_zlabel(f'Linear Discriminant 3 ({lda_columns[2]})')
                # Create a legend for the classes
                legend1 = ax.legend(*scatter.legend_elements(), title=target_col_name)
                ax.add_artist(legend1)
                plt.tight_layout()
                save_plot("lda_scatter_3d.png")
            except Exception as e:
                print(f"Error occurred while plotting/saving LDA 3D scatter plot: {e}")
                plt.close(fig) # Close the figure if plotting failed

        print("\nLDA application completed.")
        return df_lda, lda_model

    except Exception as e:
        print(f"An error occurred during LDA application: {e}")
        traceback.print_exc() # Print full traceback for debugging
        return None, None

def part2():
    """
    Executes the feature extraction steps defined in Part 2 of the project.
    - Prepares the data (X, y) from the preprocessed DataFrame 'df'.
    - Applies PCA (Step 6) and stores the result in 'df_pca_result'.
    - Applies LDA (Step 7) and stores the result in 'df_lda_result'.
    - Ensures three data representations are available: Raw (df), PCA (df_pca_result), LDA (df_lda_result).
    """
    print("\n" + "="*10 + " Part 2: Feature Extraction (PCA & LDA) " + "="*10)
    global df, df_pca_result, df_lda_result
    if df is None:
        print("Error: Processed DataFrame 'df' not found for Part 2. Run Part 1 first.")
        return

    # Use the preprocessed data from Part 1
    df_processed = df.copy()
    target_col = 'Class'

    # Basic checks before proceeding
    if target_col not in df_processed.columns:
        print(f"Error: Target column '{target_col}' not found!")
        return
    if not pd.api.types.is_numeric_dtype(df_processed[target_col]):
        print(f"Error: Target column '{target_col}' is not numerical.")
        return

    # Separate features (X) and target (y)
    X = df_processed.drop(target_col, axis=1)
    y = df_processed[target_col]

    # Ensure all features are numerical before PCA/LDA
    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
         print("Error: Feature matrix (X) is not entirely numerical.")
         non_numeric_cols = X.select_dtypes(exclude=np.number).columns.tolist()
         print(f"Non-numerical columns found: {non_numeric_cols}")
         return

    print(f"\nData prepared for Part 2: X={X.shape}, y={y.shape}")

    # Apply PCA (Step 6)
    df_pca_result, _, _ = apply_pca(X_data=X, y_data=y, target_col_name=target_col, random_state=42)
    # Apply LDA (Step 7) - using 3 components as requested
    df_lda_result, _ = apply_lda(X_data=X, y_data=y, target_col_name=target_col, n_components_lda=3)

    # Summarize the results of Part 2
    print("\n" + "="*10 + " Part 2 Completed: Results " + "="*10)
    print(f"1. Raw Data: 'df' (Shape: {df.shape})") # Preprocessed data from Part 1
    print(f"2. PCA Data: 'df_pca_result' (Shape: {df_pca_result.shape if df_pca_result is not None else 'Not Generated'})")
    print(f"3. LDA Data: 'df_lda_result' (Shape: {df_lda_result.shape if df_lda_result is not None else 'Not Generated'})")

# --- Part 3: Modeling and Evaluation Functions ---

def prepare_data(dataframe: DataFrame, target_col: str):
    """
    Utility function to separate features (X) and target (y) from a given DataFrame.
    Performs basic validation checks.
    """
    if dataframe is None or not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        print("Error: Invalid or empty DataFrame passed to prepare_data.")
        return None, None
    if target_col not in dataframe.columns:
        print(f"Error: Target column '{target_col}' not found in DataFrame.")
        return None, None

    X = dataframe.drop(target_col, axis=1)
    y = dataframe[target_col]
    print(f"Data prepared: X shape {X.shape}, y shape {y.shape}")
    return X, y

def get_classifiers_and_grids():
    """
    Defines the classifiers and their corresponding hyperparameter grids for GridSearchCV.
    Corresponds to Project Step 9 (classifiers) and Step 8 (hyperparameter tuning).
    """
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, solver='liblinear'), # Liblinear good for L1
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1), # Use all available CPU cores
        "XGBoost": xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=42, n_jobs=-1), # Specify objective and metric for multi-class
        "Naive Bayes": GaussianNB() # GaussianNB typically doesn't require hyperparameter tuning
    }
    # Define parameter grids for hyperparameter tuning in the inner loop of Nested CV
    param_grids = {
        "Logistic Regression": {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}, # Regularization strength and type
        "Decision Tree": {'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 3, 5]}, # Tree complexity parameters
        "Random Forest": {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 3], 'criterion': ['gini', 'entropy']}, # Forest size and tree complexity
        "XGBoost": {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1, 0.2], 'max_depth': [3, 5, 7], 'gamma': [0, 0.1], 'subsample': [0.8, 1.0]}, # Key XGBoost parameters
        "Naive Bayes": {} # No parameters to tune for GaussianNB in this setup
    }
    return classifiers, param_grids

def perform_nested_cv(X, y, classifier_name, classifier, param_grid, outer_k=5, inner_k=3, random_state_base=42):
    """
    Performs Nested Cross-Validation as specified in Project Step 8.
    - Outer loop (StratifiedKFold, outer_k=5 folds) for model evaluation.
    - Inner loop (StratifiedKFold, inner_k=3 folds) used by GridSearchCV for hyperparameter tuning.
    - Uses 'f1_weighted' as the scoring metric for hyperparameter selection (inner loop).
    - Calculates Accuracy, Precision, Recall, F1 (weighted) for each outer fold.
    - Returns a list of scores for each outer fold and data about the best performing fold (model, test data, score).
    - Handles potential errors during GridSearchCV gracefully.
    - Ensures different random states for outer folds and inner CV splits for robustness.
    """
    print(f"\n--- Starting Nested CV for: {classifier_name} ---")
    outer_scores = [] # Store metrics for each outer fold
    best_fold_data = {'score': -1} # Track the best model/data from the outer loop based on F1 score
    # Outer loop setup (5-fold Stratified CV)
    outer_cv = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=random_state_base)
    fold_num = 0

    # Reset index to ensure iloc works correctly with CV splits
    if isinstance(X, pd.DataFrame): X = X.reset_index(drop=True)
    if isinstance(y, pd.Series): y = y.reset_index(drop=True)

    # Outer Cross-Validation Loop
    for train_outer_idx, test_outer_idx in outer_cv.split(X, y):
        fold_num += 1
        fold_start_time = time()
        # Use a different random state for each outer fold's inner CV for robustness
        current_fold_random_state = random_state_base + fold_num
        print(f"  Outer Fold {fold_num}/{outer_k} (random_state={current_fold_random_state})...")

        # Split data for the current outer fold
        X_train_outer, X_test_outer = X.iloc[train_outer_idx], X.iloc[test_outer_idx]
        y_train_outer, y_test_outer = y.iloc[train_outer_idx], y.iloc[test_outer_idx]

        # Inner loop setup (3-fold Stratified CV for hyperparameter tuning)
        inner_cv = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=current_fold_random_state + 100) # Different seed for inner CV
        best_estimator, best_params = None, {}

        # Perform hyperparameter tuning using GridSearchCV if a parameter grid is provided
        if param_grid:
            # Use GridSearchCV for the inner loop
            grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=inner_cv, scoring='f1_weighted', n_jobs=-1, refit=True)
            try:
                grid_search.fit(X_train_outer, y_train_outer)
                best_estimator = grid_search.best_estimator_ # The model refit on the whole outer train set with best params
                best_params = grid_search.best_params_
                print(f"    Best Params (Inner CV - Fold {fold_num}): {best_params}")
                print(f"    Best Score (Inner CV - f1_weighted): {grid_search.best_score_:.4f}")
            except Exception as e:
                # Graceful handling if GridSearchCV fails (e.g., incompatible parameters)
                print(f"    ERROR: GridSearchCV failed (Fold {fold_num}): {e}")
                print(f"    Attempting to fit with default parameters...")
                try:
                    # Clone the original classifier to avoid modifying it
                    from sklearn.base import clone
                    default_estimator = clone(classifier)
                    best_estimator = default_estimator.fit(X_train_outer, y_train_outer)
                    best_params = {"status": "GridSearch Failed, Default Used"}
                    print("    Successfully fitted with default parameters.")
                except Exception as fit_e:
                    print(f"    ERROR: Fitting with default parameters also failed: {fit_e}")
                    best_estimator = None # Mark as failed
                    best_params = {"status": "GridSearch and Default Fit Failed"}
        else:
            # If no parameter grid (e.g., Naive Bayes), just fit the model
            print("    Parameter grid is empty, skipping GridSearchCV.")
            try:
                best_estimator = classifier.fit(X_train_outer, y_train_outer)
                best_params = {"status": "No GridSearch"}
            except Exception as fit_e:
                 print(f"    ERROR: Fitting model failed: {fit_e}")
                 best_estimator = None
                 best_params = {"status": "Default Fit Failed"}

        # Evaluate the best estimator found (from inner loop) on the outer test set
        if best_estimator:
            y_pred_outer = best_estimator.predict(X_test_outer)
            # Calculate performance metrics for this outer fold (Project Step 10)
            accuracy = accuracy_score(y_test_outer, y_pred_outer)
            precision = precision_score(y_test_outer, y_pred_outer, average='weighted', zero_division=0) # Weighted average for multiclass
            recall = recall_score(y_test_outer, y_pred_outer, average='weighted', zero_division=0)
            f1 = f1_score(y_test_outer, y_pred_outer, average='weighted', zero_division=0)
            fold_scores = {'Fold': fold_num, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1, 'Best Params': best_params}
            outer_scores.append(fold_scores)
            print(f"    Outer Fold {fold_num} Scores: Acc={accuracy:.4f}, F1={f1:.4f}")

            # Keep track of the overall best fold based on F1 score (for ROC curve generation)
            if f1 > best_fold_data['score']:
                print(f"    *** New best fold found (Fold {fold_num}, F1={f1:.4f}) ***")
                best_fold_data = {'model': best_estimator, 'X_test': X_test_outer, 'y_test': y_test_outer, 'score': f1, 'fold_num': fold_num}
        else:
            # Handle case where model training failed completely in this fold
            print(f"    Warning: Model could not be trained/selected for Outer Fold {fold_num}.")

        fold_end_time = time()
        print(f"    Outer Fold {fold_num} completed. Duration: {fold_end_time - fold_start_time:.2f} seconds")

    print(f"--- Nested CV Completed for: {classifier_name} ---")
    # Return scores from all outer folds and data from the single best fold
    return outer_scores, best_fold_data

def plot_roc_ovr(model, X_test, y_test, n_classes, classifier_name, data_name, class_names=None):
    """
    Plots One-vs-Rest (OvR) ROC curves for a multi-class classification problem (Project Step 11).
    - Uses the results from the single best outer fold identified in nested CV.
    - Requires the model to have a 'predict_proba' method.
    - Binarizes the output and calculates ROC/AUC for each class against the rest.
    - Plots all class ROC curves on a single graph.
    - Saves the plot to a file.
    - Returns a dictionary of ROC AUC scores per class.
    """
    print(f"\n--- Plotting ROC Curves for: {classifier_name} ({data_name}) ---")
    # Check if the model can provide probability estimates
    if not hasattr(model, "predict_proba"):
        print(f"Warning: Classifier '{classifier_name}' does not have 'predict_proba' method. Cannot plot ROC.")
        return None
    # Ensure class names are provided or generate default ones
    if class_names is None or len(class_names) != n_classes:
        class_names = [f'Class {i}' for i in range(n_classes)] # Default names if not provided

    try:
        # Binarize the true labels for OvR ROC calculation
        y_test_binarized = label_binarize(y_test, classes=range(n_classes))
        # Get probability scores for each class
        y_score = model.predict_proba(X_test)

        # Sanity check: Ensure predict_proba output matches the number of classes
        if y_score.shape[1] != n_classes:
             print(f"Warning: Output shape of predict_proba ({y_score.shape[1]}) does not match number of classes ({n_classes}). Cannot plot ROC.")
             # Handle common issue in binary classification where predict_proba returns only one column
             if n_classes == 2 and y_score.shape[1] == 1:
                 print("Adjusting for binary classification predict_proba output.")
                 y_score_adjusted = np.zeros((y_score.shape[0], 2))
                 y_score_adjusted[:, 1] = y_score[:, 0] # Probability of class 1
                 y_score_adjusted[:, 0] = 1 - y_score[:, 0] # Probability of class 0
                 y_score = y_score_adjusted
             # Handle case where predict_proba might return extra columns (less common)
             elif n_classes == 2 and y_score.shape[1] > 2:
                 print("Adjusting multi-output predict_proba for binary case.")
                 y_score = y_score[:, :2] # Take only the first two columns
             else:
                 return None # Cannot resolve shape mismatch

        # Compute ROC curve and ROC area for each class
        fpr, tpr, roc_auc = dict(), dict(), dict()
        for i in range(n_classes):
            try:
                fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            except ValueError as ve:
                # Handle cases where ROC cannot be computed (e.g., only one class present in test fold for this OvR split)
                print(f"Warning: Could not compute ROC for class {i} ({class_names[i]}): {ve}. Setting AUC to NaN.")
                fpr[i], tpr[i], roc_auc[i] = np.array([0, 1]), np.array([0, 1]), np.nan # Assign dummy values and NaN AUC

        # Plot all ROC curves
        plt.figure(figsize=(10, 8))
        # Use a colormap for distinct class colors
        try:
            cmap = matplotlib.colormaps['viridis'] # Newer matplotlib versions
        except AttributeError:
            cmap = plt.get_cmap('viridis') # Older matplotlib versions
        colors = cmap(np.linspace(0, 1, n_classes))
        valid_auc_count = 0 # Track if any AUC could be computed

        for i, color in enumerate(colors):
            class_label = class_names[i]
            if not np.isnan(roc_auc[i]): # Only plot if AUC is valid
                plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'{class_label} (AUC = {roc_auc[i]:.3f})')
                valid_auc_count += 1
            else: # Still add label for completeness, indicating NaN AUC
                plt.plot([], [], color=color, lw=2, label=f'{class_label} (AUC = NaN)')

        # Plot the random chance line
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Chance (AUC = 0.5)')
        # Set plot limits and labels
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Recall / Sensitivity)')
        plt.title(f'ROC Curves (One-vs-Rest) - {classifier_name} ({data_name})')
        if valid_auc_count > 0:
            plt.legend(loc="lower right")
        else: # Modify legend if no valid AUCs were computed
            plt.legend(title="AUC could not be computed", loc="lower right")
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        safe_clf_name = classifier_name.replace(" ", "_") # Make filename safe
        filename = f"roc_curve_{data_name}_{safe_clf_name}.png"
        save_plot(filename)

        # Print AUC scores
        print("ROC AUC Scores:")
        avg_auc = np.nanmean([roc_auc.get(i, np.nan) for i in range(n_classes)]) # Calculate average AUC, ignoring NaNs
        for i in range(n_classes):
            class_label = class_names[i]
            auc_score = roc_auc.get(i, np.nan)
            auc_score_str = f"{auc_score:.4f}" if not np.isnan(auc_score) else "NaN"
            print(f"  {class_label}: {auc_score_str}")
        if not np.isnan(avg_auc):
            print(f"  Average AUC (excluding NaN): {avg_auc:.4f}")
        else:
            print("  Average AUC: Could not be computed")
        return roc_auc # Return the dictionary of AUC scores

    except Exception as e:
        print(f"Error plotting/saving ROC curves: {e}")
        traceback.print_exc() # Print full traceback
        plt.close() # Ensure plot is closed on error
        return None


def plot_comparison_matrices(results_df):
    """
    Generates comparison visualizations based on the aggregated results from Nested CV.
    - Creates heatmaps comparing F1 Score and Accuracy across classifiers and data representations.
    - Creates a bar plot comparing F1 scores.
    - Prints the best performing model combination based on mean F1 score.
    - Saves the plots and a summary table (CSV) to the 'outputs' folder.
    """
    try:
        print("\n--- Generating Comparison Matrices ---")

        # F1 Score Heatmap
        plt.figure(figsize=(14, 10))
        # Pivot the results table for the heatmap
        comparison_pivot_f1 = results_df.pivot(index='Classifier', columns='Data', values='F1 Score Mean')
        sns.heatmap(comparison_pivot_f1, annot=True, cmap='viridis', fmt='.4f', linewidths=.5)
        plt.title('F1 Score Comparison Matrix (Classifier vs Data Representation)')
        plt.tight_layout()
        save_plot("comparison_matrix_f1.png")

        # Accuracy Heatmap
        plt.figure(figsize=(14, 10))
        comparison_pivot_acc = results_df.pivot(index='Classifier', columns='Data', values='Accuracy Mean')
        sns.heatmap(comparison_pivot_acc, annot=True, cmap='viridis', fmt='.4f', linewidths=.5)
        plt.title('Accuracy Comparison Matrix (Classifier vs Data Representation)')
        plt.tight_layout()
        save_plot("comparison_matrix_accuracy.png")

        # Identify and print the best model based on mean F1 score
        best_model_idx = results_df['F1 Score Mean'].idxmax()
        best_model = results_df.loc[best_model_idx]
        print(f"\nBest Model Performance (based on F1 Score Mean):")
        print(f"  Data Representation: {best_model['Data']}")
        print(f"  Classifier: {best_model['Classifier']}")
        print(f"  F1 Score: {best_model['F1 Score Mean']:.4f} ± {best_model['F1 Score Std']:.4f}")
        print(f"  Accuracy: {best_model['Accuracy Mean']:.4f} ± {best_model['Accuracy Std']:.4f}")

        # F1 Score Bar Plot Comparison
        plt.figure(figsize=(16, 10))
        sns.barplot(x='Classifier', y='F1 Score Mean', hue='Data', data=results_df)
        plt.title('F1 Score Comparison by Classifier and Data Representation')
        plt.ylim(0, 1) # F1 score is between 0 and 1
        plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
        plt.tight_layout()
        save_plot("comparison_barplot_f1.png")

        # Create and save a summary table (CSV)
        summary_table = results_df.pivot_table(
            index='Classifier',
            columns='Data',
            values=['F1 Score Mean', 'Accuracy Mean'], # Include key metrics
            aggfunc='mean' # Should already be means, but ensures aggregation if needed
        ).round(4)

        summary_filename = "model_comparison_table.csv"
        # Determine output path relative to script location
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = os.getcwd()
        project_dir = os.path.dirname(script_dir)
        output_path = os.path.join(project_dir, "outputs", summary_filename)
        summary_table.to_csv(output_path)
        print(f"\nComparison table saved to '{output_path}'.")

        print("\nComparison matrices generated successfully.")
    except Exception as e:
        print(f"Error generating comparison matrices: {e}")
        traceback.print_exc()

def part3():
    """
    Executes the modeling and evaluation steps defined in Part 3 of the project.
    - Iterates through the three data representations (Raw, PCA, LDA).
    - For each data representation, iterates through the specified classifiers.
    - Performs Nested Cross-Validation (Step 8) for each combination.
    - Collects performance metrics (mean and std dev from outer folds - Step 10).
    - Plots ROC curves based on the best outer fold result for each combination (Step 11).
    - Aggregates all results and generates comparison plots/tables.
    """
    print("\n" + "="*20 + " Part 3: Modeling and Evaluation " + "="*20)
    global df, df_pca_result, df_lda_result, le # Access the data representations and label encoder

    # Prepare dictionary of datasets to iterate over
    datasets = {}
    if df is not None: datasets["Raw"] = df
    else: print("Warning: Raw data (df) not found.")
    if df_pca_result is not None: datasets["PCA"] = df_pca_result
    else: print("Warning: PCA data not found.")
    if df_lda_result is not None: datasets["LDA"] = df_lda_result
    else: print("Warning: LDA data not found.")

    if not datasets:
        print("Error: No datasets available for processing.")
        return

    target_col = 'Class'
    n_classes = 0
    class_names = []
    # Use the raw dataset (or the first available one) to determine class info consistently
    base_df_for_classes = datasets.get("Raw", next(iter(datasets.values())) if datasets else None)

    # Determine number of classes and their names (using the LabelEncoder if available)
    if base_df_for_classes is not None and target_col in base_df_for_classes.columns:
        target_series = base_df_for_classes[target_col]
        if pd.api.types.is_numeric_dtype(target_series):
            n_classes = target_series.nunique() # Get number of unique classes
            unique_values = sorted(target_series.unique()) # Get the unique numerical labels
            # Try to get original class names from the LabelEncoder fitted in Part 1
            if le is not None and hasattr(le, 'classes_') and len(le.classes_) == n_classes:
                 try:
                     # Map numerical labels back to original names
                     class_names = list(le.inverse_transform(unique_values))
                     print(f"Class names obtained from LabelEncoder: {class_names}")
                 except ValueError:
                     # Handle case where current labels might not match encoder's known labels
                     print("Warning: LabelEncoder could not transform current values. Using numerical names.")
                     class_names = [str(val) for val in unique_values]
            else:
                # Fallback if LabelEncoder is missing or inconsistent
                class_names = [str(val) for val in unique_values]
                print(f"Warning: LabelEncoder information missing or inconsistent. Using numerical values as class names: {class_names}")
        else:
            print(f"Error: Target column ('{target_col}') is not numerical in the base dataset.")
            return
    else:
        print(f"Error: Could not retrieve class information. Base dataset or target column missing/invalid.")
        return

    print(f"Total Number of Classes: {n_classes}")

    # Get classifiers and parameter grids
    classifiers, param_grids = get_classifiers_and_grids()
    all_results = [] # List to store results from all combinations

    # --- Main Loop: Iterate through Datasets and Classifiers ---
    for data_name, current_df in datasets.items():
        print(f"\n{'='*15} Processing Dataset: {data_name} {'='*15}")
        # Prepare X and y for the current dataset
        X, y = prepare_data(current_df, target_col)
        if X is None or y is None or X.empty or y.empty:
            print(f"Skipping dataset '{data_name}' due to preparation error.")
            continue # Skip to the next dataset if preparation fails

        # Iterate through each classifier
        for clf_name, classifier in classifiers.items():
            start_clf_time = time()
            print(f"\n--- Classifier: {clf_name} ({data_name}) ---")
            param_grid = param_grids.get(clf_name, {}) # Get hyperparameter grid

            # Perform Nested Cross-Validation (Step 8)
            outer_scores, best_fold_data = perform_nested_cv(X, y, clf_name, classifier, param_grid, random_state_base=42)

            # Process results if Nested CV was successful
            if outer_scores:
                df_scores = pd.DataFrame(outer_scores) # Convert fold scores to DataFrame
                # Calculate mean and std dev of metrics across outer folds (Step 10)
                results = {
                    'Data': data_name, 'Classifier': clf_name,
                    'F1 Score Mean': df_scores['F1 Score'].mean(), 'F1 Score Std': df_scores['F1 Score'].std(),
                    'Accuracy Mean': df_scores['Accuracy'].mean(), 'Accuracy Std': df_scores['Accuracy'].std(),
                    'Precision Mean': df_scores['Precision'].mean(), 'Precision Std': df_scores['Precision'].std(),
                    'Recall Mean': df_scores['Recall'].mean(), 'Recall Std': df_scores['Recall'].std(),
                    'Num Folds': len(df_scores), # Number of successful outer folds
                    'Best Fold Num': best_fold_data.get('fold_num', 'N/A'), # Which fold was best
                    'Best Fold F1': best_fold_data.get('score', -1) # F1 score of the best fold
                }
                all_results.append(results) # Add results to the overall list
                # Print summary for this classifier/dataset combination
                print(f"\nAverage Scores ({clf_name} - {data_name}):")
                print(f"  F1 Score: {results['F1 Score Mean']:.4f} +/- {results['F1 Score Std']:.4f}")
                print(f"  Accuracy: {results['Accuracy Mean']:.4f} +/- {results['Accuracy Std']:.4f}")

                # Plot ROC curve using the best model from the best fold (Step 11)
                if best_fold_data.get('score', -1) > -1 and hasattr(best_fold_data.get('model'), 'predict_proba'):
                    plot_roc_ovr(model=best_fold_data['model'], X_test=best_fold_data['X_test'],
                                 y_test=best_fold_data['y_test'], n_classes=n_classes,
                                 classifier_name=clf_name, data_name=data_name, class_names=class_names)
                elif best_fold_data.get('score', -1) <= -1:
                    print(f"Warning: No valid best fold found for {clf_name} ({data_name}), cannot save ROC.")
                else: # Model exists but doesn't have predict_proba
                     print(f"Warning: Best model for {clf_name} ({data_name}) does not support predict_proba, cannot save ROC.")

            else:
                 # Handle case where nested CV failed to produce any scores
                 print(f"Warning: No results recorded for {clf_name} ({data_name}).")

            end_clf_time = time()
            print(f"--- {clf_name} ({data_name}) completed. Total Time: {end_clf_time - start_clf_time:.2f} seconds ---")
    # --- End of Main Loop ---

    # --- Final Summary and Comparison ---
    print("\n" + "="*20 + " Summary of All Results " + "="*20)
    if all_results:
        # Create a DataFrame from the collected results
        results_df = pd.DataFrame(all_results).round(4)
        # Define column order for better readability
        cols_order = ['Data', 'Classifier', 'F1 Score Mean', 'F1 Score Std', 'Accuracy Mean', 'Accuracy Std',
                      'Precision Mean', 'Precision Std', 'Recall Mean', 'Recall Std', 'Num Folds', 'Best Fold Num', 'Best Fold F1']
        # Reorder columns, handling potential missing columns if a step failed entirely
        results_df = results_df[[col for col in cols_order if col in results_df.columns]]
        print(results_df.to_string()) # Print the full summary table

        # Generate comparison plots and save summary table
        plot_comparison_matrices(results_df)
        # Save the detailed results DataFrame to CSV
        try:
            results_filename = "modeling_results_summary.csv"
            # Determine output path
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
            except NameError:
                script_dir = os.getcwd()
            project_dir = os.path.dirname(script_dir)
            output_path = os.path.join(project_dir, "outputs", results_filename)
            results_df.to_csv(output_path, index=False)
            print(f"\nResults summary saved to '{output_path}'.")
        except Exception as e:
            print(f"\nError saving results summary CSV: {e}")
    else:
        print("No modeling results were generated.")

def main():
    """
    Main function to run the entire machine learning pipeline:
    Part 1 (Preprocessing), Part 2 (Feature Extraction), Part 3 (Modeling/Evaluation).
    Measures and prints the total execution time.
    """
    start_time = time() # Start timer
    part1() # Execute data preprocessing
    part2() # Execute feature extraction (PCA, LDA)
    part3() # Execute modeling, evaluation, and comparison
    end_time = time() # End timer
    print(f"\n=== All Operations Completed ===")
    print(f"Total Elapsed Time: {(end_time - start_time) / 60:.2f} minutes")

if __name__ == "__main__":
    main()