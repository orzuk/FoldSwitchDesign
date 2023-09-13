# Some utilities for proteins and their mutations
import pandas as pd


def genetic_code():
    """Return the standard genetic code as a dictionary."""
    code = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
    }
    return code


# Get all the possible amino acids that we get with a single point mutation
# for a specific codon
def point_mutation_to_aa(codon, genetic_code_dict):
    aa_mut_list = [genetic_code_dict.get(codon)] # this is the current aa
    for pos in range(3):
        for s in ["A", "C", "G", "T"]:
            aa_mut_list.append(genetic_code_dict.get(codon[:pos] + s + codon[pos+1:]))
    return list(set(aa_mut_list))


# Get all codons that code for a specific amino-acid
def aa_to_codon(aa, genetic_code_dict):
    return [x for x in genetic_code_dict if genetic_code_dict[x] == aa]


# Get all aa that can be obtained by a single point mutation from a given aa,
# where we don't know the codon
def aa_point_mutation_to_aa(aa, genetic_code_dict):
    return list(set([j for l in [point_mutation_to_aa(c, genetic_code_dict) for c in aa_to_codon(aa, genetic_code_dict)] for j in l]))


# Run design on every position, collect
def design_every_position(S, pdbID):
    n = len(S) # get number of amino-acids
    S_des = ['']*n # The output designed sequence
    for i in range(n):  # loop on positions
        cur_S = run_design(S, pdbID, i)  # run design function using ProteinMPNN, keeping all positions fixed except i
        S_des[i] = cur_S[i]  # change the i-th letter of the i-th protein
    S_des = "".join(S_des) # convert to string
    return S_des


# Run design to both A and B, then fold both of them with alpha-fold. See if we get something closer
def compare_designs(S, pdbID1, pdbID2):
    S1 = design_every_position(S, pdbID1)
    S2 = design_every_position(S, pdbID2)

    # Compare sequences:
    print("\nS: ", S, "\nS1: ", S1, "\nS2: ", S2)

    AF = Alphafold(S)  # Fold the natural sequence
    AF1 = Alphafold(S1)  # Fold the first one
    AF2 = Alphafold(S2)  # Fold the second one

    TM = ['TM1', 'TM2']
    SEQ = ['S', 'S1', 'S2']
    df_tm = pd.DataFrame(columns=TM, index=SEQ)
    S_true_list = [pdbID1, pdbID2]
    S_pred_list = [AF, AF1, AF2]
    for i_true in range(2):   # loop on the two true structures
        for j_pred in range(3):  # S_predicted in [AF, AF1, AF2]:  # loop on the three predicted structures
            df_tm[TM[i_true]][SEQ[j_pred]] = TMScore(S_true_list[i_true], S_pred_list[j_pred])  # Compute TMScore similarity

    print(df_tm)
    return df_tm, S, S1, S2, AF, AF1, AF2  # Return all sequences, structures and their similarity

# Example usage:
print("Hello")
genetic_code_dict = genetic_code()
aa_list = list(set(genetic_code_dict.values())) # all possible amino-acids
aa_list.remove("*")
codon_list = list(genetic_code_dict)
codon = 'GGA'
amino_acid = genetic_code_dict.get(codon, 'Unknown')
print(f'The amino acid corresponding to {codon} is {amino_acid}')

for c in codon_list:
    print(c, genetic_code_dict.get(c), point_mutation_to_aa(c, genetic_code_dict))

for aa in aa_list:
    print(aa,  aa_point_mutation_to_aa(aa, genetic_code_dict))



