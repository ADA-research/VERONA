# Copyright 2025 ADA Reseach Group and VERONA council. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import pandas as pd
import argparse
import glob
import os
#check for one network whether all correctly classified images are there

#load CSV and save indices of verified indices in results_df
def check_verified_instances(model_name, attack_method,indice,batch_size):
    pattern = f"/scratch-shared/abosman/data/results_LION/{model_name}-{attack_method}--{indice}*"
    matches = glob.glob(pattern)
    
    if not matches:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")

    latest_file = max(matches, key=os.path.getmtime)

    print(f"Loading: {latest_file}")
    df = pd.read_csv(latest_file)
    

        

#check whether all the indices are in the temp files

#do something if they are in one but not the other

#add only not verified instances + correctly classified to the verification loop




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name of the network from timm we want to use')
    parser.add_argument('--attack_method', type=str, help = 'Name of the method we are using for finding adversarial examples.')
    parser.add_argument('--indices', type=int, help ='indice where to start the search.')
    parser.add_argument('--batch_size', type=int, help ='indice where to start the search.')
    args = parser.parse_args()
    
    model_name = args.model_name
    attack_method = args.attack_method
    indice =  args.indices
    batch_size = args.batch_size
    print(model_name)
    
    check_verified_instances(model_name, attack_method,indice,batch_size)

    