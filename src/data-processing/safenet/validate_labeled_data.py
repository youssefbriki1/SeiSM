import pickle, numpy as np

with open('data/testing_labels.pickle', 'rb') as f:
    ours = pickle.load(f)
with open('data/labels_mag_C_class4_for_eval_10y_in_11_16.pickle', 'rb') as f:
    ref = pickle.load(f)

print(f'Ours: {len(ours)} entries, each shape {ours[0].shape}')
print(f'Ref:  {len(ref)} entries, each shape {ref[0].shape}')
print()

total_match = 0
total = 0

for i in range(10):
    our_arr = ours[i] 
    ref_arr = ref[i]
    
    matches = (our_arr == ref_arr).sum()
    total_match += matches
    total += len(ref_arr)
    
    mismatches = np.where(our_arr != ref_arr)[0]
    label_yr = 2012 + i
    
    if len(mismatches) == 0:
        print(f'  Entry [{i}] label year {label_yr}: {matches}/85 PERFECT MATCH ✓')
    else:
        print(f'  Entry [{i}] label year {label_yr}: {matches}/85 match, {len(mismatches)} differ')
        for idx in mismatches:
            print(f'    Patch {idx+1}: ours={our_arr[idx]} ref={ref_arr[idx]}')

print(f'\nTotal: {total_match}/{total} ({100*total_match/total:.1f}%)')
