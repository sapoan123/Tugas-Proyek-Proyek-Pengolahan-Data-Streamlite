"""
MAIN PIPELINE ORCHESTRATOR
Mengintegrasikan seluruh workflow: EDA → Prepare → Train → Evaluate
"""

import os
import sys
import time
from datetime import datetime

# Import semua modul pipeline
import eda
import prepare
import train
import evaluate


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  🔹 {text}")
    print("="*70 + "\n")


def print_footer(text):
    """Print formatted footer"""
    print("\n" + "="*70)
    print(f"   {text}")
    print("="*70 + "\n")


def run_phase_1_eda():
    """PHASE 1: Exploratory Data Analysis"""
    print_header("PHASE 1: EXPLORATORY DATA ANALYSIS (EDA)")
    
    try:
        print(" Loading raw data...")
        df = eda.load_data()
        print(f" Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        
        print("\n Performing EDA...")
        eda.perform_eda(df)
        
        print_footer("EDA SELESAI")
        return df
    
    except Exception as e:
        print(f" Error di Phase 1 (EDA): {e}")
        sys.exit(1)


def run_phase_2_prepare(df):
    """PHASE 2: Data Preparation & Preprocessing"""
    print_header("PHASE 2: DATA PREPARATION & PREPROCESSING")
    
    try:
        print(" Preprocessing data...")
        df_prepared = prepare.preprocess_data(df)
        print(f" Data prepared: {df_prepared.shape[0]} rows × {df_prepared.shape[1]} columns")
        
        print("\n Saving prepared data...")
        prepare.save_data(df_prepared)
        
        print_footer("DATA PREPARATION SELESAI")
        return df_prepared
    
    except Exception as e:
        print(f" Error di Phase 2 (Prepare): {e}")
        sys.exit(1)


def run_phase_3_train(df_prepared):
    """PHASE 3: Model Training (KMeans + ARM)"""
    print_header("PHASE 3: MODEL TRAINING (KMEANS + ARM)")
    
    try:
        print(" Starting training pipeline...")
        train.main()
        
        print_footer("MODEL TRAINING SELESAI")
    
    except Exception as e:
        print(f" Error di Phase 3 (Train): {e}")
        sys.exit(1)


def run_phase_4_evaluate():
    """PHASE 4: Model Evaluation & Metrics"""
    print_header("PHASE 4: MODEL EVALUATION")
    
    try:
        print(" Loading trained data and rules...")
        df, rules = evaluate.load_data()
        
        print("\n Evaluating KMeans clustering...")
        kmeans_results = evaluate.evaluate_kmeans(df)
        
        print("\n Evaluating Association Rules...")
        arm_results = evaluate.evaluate_apriori(rules)
        
        print_footer("EVALUATION SELESAI")
    
    except Exception as e:
        print(f" Error di Phase 4 (Evaluate): {e}")
        sys.exit(1)


def main():
    """Main pipeline orchestrator"""
    print("\n" + " "*35)
    print("  MACHINE LEARNING PIPELINE - FULL ORCHESTRATION")
    print("  Data Mining Project: Online Retail Clustering & Association Rules")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(" "*35 + "\n")
    
    start_time = time.time()
    
    # ===== PHASE 1: EDA =====
    df = run_phase_1_eda()
    
    # ===== PHASE 2: PREPARE =====
    df_prepared = run_phase_2_prepare(df)
    
    # ===== PHASE 3: TRAIN =====
    run_phase_3_train(df_prepared)
    
    # ===== PHASE 4: EVALUATE =====
    run_phase_4_evaluate()
    
    # ===== FINAL SUMMARY =====
    elapsed_time = time.time() - start_time
    
    print("\n" + " "*35)
    print("   PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"  Total Execution Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print("="*70)
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n Output files saved in:")
    print("   - plots/           (Visualizations)")
    print("   - processed/       (Prepared data)")
    print("   - models/          (Trained models)")
    print("   - metrics.json     (Evaluation results)")
    print("   - training_results.json  (Training statistics)")
    print(" "*35 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n Unexpected error: {e}")
        sys.exit(1)
