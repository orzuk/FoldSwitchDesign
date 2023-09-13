# Some utilities for proteins and their mutations

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
