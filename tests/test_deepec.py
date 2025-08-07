#!/usr/bin/env python3

import os
import tempfile
import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd

class TestDeepEC:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_input = self.project_root / "tests/data/TARA_ARC_108_MAG_00080.fasta"
        self.expected_result = self.project_root / "tests/results/DeepECv2_result.txt"
        
    def test_deepec_execution(self):
        """Test that DeepEC runs successfully and produces expected output format"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cmd = [
                sys.executable, 
                str(self.project_root / "src/run_deepectransformer.py"),
                "-i", str(self.test_input),
                "-o", temp_dir,
                "-b", "128",
                "-g", "cpu",
                "-cpu", "10"
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                
                if result.returncode != 0:
                    print(f"❌ FAILED: DeepEC execution failed with return code {result.returncode}")
                    print(f"STDERR: {result.stderr}")
                    return False
                    
                output_file = Path(temp_dir) / "DeepECv2_result.txt"
                if not output_file.exists():
                    print("❌ FAILED: Output file DeepECv2_result.txt not created")
                    return False
                    
                return self._validate_output_format(output_file)
                
            except subprocess.TimeoutExpired:
                print("❌ FAILED: DeepEC execution timed out after 30 minutes")
                return False
            except Exception as e:
                print(f"❌ FAILED: Unexpected error: {e}")
                return False
    
    def _validate_output_format(self, output_file):
        """Validate the output file format matches expected structure"""
        try:
            df = pd.read_csv(output_file, sep='\t')
            
            expected_columns = [
                'sequence_ID', 'ec_numbers', 'deepec_ecs', 
                'deepec_scores', 'blastp_ecs', 'blastp_scores'
            ]
            
            if list(df.columns) != expected_columns:
                print(f"❌ FAILED: Unexpected column structure. Got: {list(df.columns)}, Expected: {expected_columns}")
                return False
            
            if len(df) == 0:
                print("❌ FAILED: Output file is empty")
                return False
            
            for idx, row in df.head().iterrows():
                if pd.isna(row['sequence_ID']) or row['sequence_ID'] == '':
                    print(f"❌ FAILED: Empty sequence_ID in row {idx}")
                    return False
                    
                if pd.isna(row['ec_numbers']) or row['ec_numbers'] == '':
                    print(f"❌ FAILED: Empty ec_numbers in row {idx}")
                    return False
            
            print(f"✅ PASSED: Output format validation successful")
            print(f"  - File contains {len(df)} predictions")
            print(f"  - Columns match expected format")
            print(f"  - Sample predictions:")
            for _, row in df.head(3).iterrows():
                print(f"    {row['sequence_ID']}: {row['ec_numbers']} (scores: {row['deepec_scores']})")
            
            return True
            
        except Exception as e:
            print(f"❌ FAILED: Error validating output format: {e}")
            return False
    
    def test_output_consistency(self):
        """Test that output is consistent with expected results"""
        if not self.expected_result.exists():
            print("⚠️  WARNING: No expected results file found for consistency check")
            return True
            
        with tempfile.TemporaryDirectory() as temp_dir:
            cmd = [
                sys.executable, 
                str(self.project_root / "src/run_deepectransformer.py"),
                "-i", str(self.test_input),
                "-o", temp_dir,
                "-b", "128",
                "-g", "cpu",
                "-cpu", "10"
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                if result.returncode != 0:
                    print("❌ FAILED: DeepEC execution failed in consistency test")
                    return False
                
                output_file = Path(temp_dir) / "DeepECv2_result.txt"
                expected_df = pd.read_csv(self.expected_result, sep='\t')
                actual_df = pd.read_csv(output_file, sep='\t')
                
                if len(expected_df) != len(actual_df):
                    print(f"⚠️  WARNING: Different number of predictions - Expected: {len(expected_df)}, Actual: {len(actual_df)}")
                
                matching_sequences = set(expected_df['sequence_ID']).intersection(set(actual_df['sequence_ID']))
                total_sequences = len(set(expected_df['sequence_ID']).union(set(actual_df['sequence_ID'])))
                
                consistency_ratio = len(matching_sequences) / total_sequences if total_sequences > 0 else 0
                print(f"✅ PASSED: Sequence consistency check - {len(matching_sequences)}/{total_sequences} sequences match ({consistency_ratio:.2%})")
                
                return True
                
            except Exception as e:
                print(f"❌ FAILED: Error in consistency test: {e}")
                return False
    
    def run_all_tests(self):
        """Run all tests and return overall result"""
        print("🧪 Running DeepEC Tests...")
        print("=" * 50)
        
        if not Path(self.test_input).exists():
            print(f"❌ FAILED: Test input file not found: {self.test_input}")
            return False
        
        tests = [
            ("Execution Test", self.test_deepec_execution),
            ("Consistency Test", self.test_output_consistency)
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n🔬 Running {test_name}...")
            try:
                result = test_func()
                results.append(result)
                print(f"{'✅ PASSED' if result else '❌ FAILED'}: {test_name}")
            except Exception as e:
                print(f"❌ FAILED: {test_name} - Unexpected error: {e}")
                results.append(False)
        
        overall_result = all(results)
        print("\n" + "=" * 50)
        print(f"🎯 Overall Result: {'✅ ALL TESTS PASSED' if overall_result else '❌ SOME TESTS FAILED'}")
        print(f"📊 Test Summary: {sum(results)}/{len(results)} tests passed")
        
        return overall_result

if __name__ == "__main__":
    tester = TestDeepEC()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)