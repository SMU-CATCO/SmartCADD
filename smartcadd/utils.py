import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from dimorphite_dl import DimorphiteDL



# get the distance between any two pharmacophores
def distance_claculator(self, num1, num2):
    d = np.linalg.norm(np.array(num1) - np.array(num2))
    return d


def acc_don_dist(zero_mids, drug_hbs):
    dhs = []
    for d_h in drug_hbs:
        dis = distance_claculator(zero_mids[0][0], d_h)
        dhs.append(dis)
    return dhs


# middle point calculated coomon to both lead and target compounds.
def find_zero_midpoints(drug_mid_points, lead_mid_points):
    zero_mids = []
    for d_m in drug_mid_points:
        for l_m in lead_mid_points:
            dist = distance_claculator(d_m, l_m)
            if 0 <= dist <= 0.5:
                zero_mids.append([d_m, l_m])
    return zero_mids


# scoring function to get the likelihood of pharmacophore location.
def scoring_function(dhs, lead_hbs, zero_mids):
    point_table = []
    value_ranges = [0.2, 0.4, 0.6, 0.8, 1.0]
    for value in value_ranges:
        point = 0
        for dis in dhs:
            for l_h in lead_hbs:
                distance_1 = distance_claculator(zero_mids[0][0], l_h)
                if (dis - value) <= distance_1 <= (dis + value):
                    point += 1
                    break
        point_table.append(point)
    return point_table


# middle point calculated coomon to both lead and target compounds.
def find_zero_midpoints(drug_mid_points, lead_mid_points):
    zero_mids = []
    for d_m in drug_mid_points:
        for l_m in lead_mid_points:
            dist = distance_claculator(d_m, l_m)
            if 0 <= dist <= 0.5:
                zero_mids.append([d_m, l_m])
    return zero_mids


def other_middle_points(drug_mid_points, lead_mid_points, zero_mids):
    point_table = []
    other_middle_d_points = []
    other_middle_l_points = []
    for d_mid in drug_mid_points:
        mid_dist_calc = distance_claculator(zero_mids[0][0], d_mid)
        if not 0 <= mid_dist_calc <= 0.5:
            other_middle_d_points.append(mid_dist_calc)

    for l_mid in lead_mid_points:
        mid_dist_calcu = distance_claculator(zero_mids[0][0], l_mid)
        if not 0 <= mid_dist_calcu <= 0.5:
            other_middle_l_points.append(mid_dist_calcu)

    value_ranges = [0.2, 0.4, 0.6, 0.8, 1.0]

    # scoring function for the other ring systems except the aligne one
    for value in value_ranges:
        if len(other_middle_d_points) == 1 and len(other_middle_l_points) == 1:
            middle_point_collector = 0
            if (
                (other_middle_d_points[0] - value)
                <= other_middle_l_points[0]
                <= (other_middle_d_points[0] + value)
            ):
                middle_point_collector += 2
            point_table.append(middle_point_collector)

        elif (
            len(other_middle_d_points) == 2 and len(other_middle_l_points) == 1
        ):
            middle_point_collector = 0
            for other_D in other_middle_d_points:
                if (
                    (other_D - value)
                    <= other_middle_l_points[0]
                    <= (other_D + value)
                ):
                    middle_point_collector += 2
            point_table.append(middle_point_collector)

        elif (
            len(other_middle_d_points) == 2 and len(other_middle_l_points) == 2
        ):
            middle_point_collector = 0
            for other_D in other_middle_d_points:
                for other_L in other_middle_l_points:
                    if (
                        (other_D - value)
                        <= other_middle_l_points[0]
                        <= (other_D + value)
                    ):
                        middle_point_collector += 2
                        break
            point_table.append(middle_point_collector)
    return point_table

# Tautomer Canonicalization
def reorderTautomers(mol):
    enumerator = rdMolStandardize.TautomerEnumerator()
    canon = enumerator.Canonicalize(mol)
    csmi = Chem.MolToSmiles(canon)
    res = [canon]
    tauts = enumerator.Enumerate(mol)
    smis = [Chem.MolToSmiles(x) for x in tauts]
    stpl = sorted((x,y) for x,y in zip(smis,tauts) if x!=csmi)
    res += [y for x,y in stpl]
    return res


# protonation modules
def get_protonation_states(smile):
    mol_list = []
    dimorphite_dl = DimorphiteDL(
    min_ph=4.5,
    max_ph=8.0,
    max_variants=128,
    label_states=False,
    pka_precision=1.0
    )
    result_smile_list = dimorphite_dl.protonate(smile)
    for smi in result_smile_list:
        mol = Chem.MolFromSmiles(smi)
        mol_list.append(mol)
    return mol_list