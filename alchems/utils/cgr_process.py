from __future__ import annotations

import copy
import traceback
from collections.abc import Iterable
from typing import Any

from chython import smiles as smiles_chython
from chython.containers import ReactionContainer
from chython.containers.bonds import DynamicBond



def route_smi_2_cgr(pathway, reverse=False): # True for AiZynthFInder, False for ASKCOS
    """Converts a pathway of SMILES strings to a list of CGRs."""
    cgr_pathway = []
    inversed_pathway = pathway[::-1] if reverse else pathway
    for reaction_str in inversed_pathway:
        reactants = []
        product = smiles_chython(reaction_str[1])
        for reactant_smiles in reaction_str[0]:
            reactant = smiles_chython(reactant_smiles)
            try:
                reactant.kekule()
            except:
                pass
            reactant.implicify_hydrogens()
            reactant.thiele()
            reactants.append(reactant)
        reaction = ReactionContainer(reactants=reactants, products = [product])
        reaction.reset_mapping(keep_reactants_numbering=False)
        cgr_pathway.append(reaction)
    return cgr_pathway

def find_remap(lst):
    """
    Given a sorted list `lst` whose true length N is known to be len(lst),
    returns a dict mapping each value > N in lst to the missing values in 1..N.

    Example:
      L = [1,2,...,18,20,21,22,23]  # len=22
      => missing = [19]
         out_of_range = [23]
      => {23: 19}
    """
    N = len(lst)
    # 1) which values in the “ideal” 1..N are missing?
    missing = sorted(set(range(1, N+1)) - set(x for x in lst if x <= N))
    # 2) which values in lst have “overflowed” past N?
    out_of_range = sorted(x for x in lst if x > N)

    if len(missing) != len(out_of_range):
        raise ValueError(f"got {len(missing)} missing slots but {len(out_of_range)} overflow values")

    # 3) pair them up in ascending order
    return dict(zip(out_of_range, missing))

def _atom_symbol(atom):
    for attr in ("symbol", "atomic_symbol", "element"):
        val = getattr(atom, attr, None)
        if not val:
            continue
        if isinstance(val, str):
            return val
        sym = getattr(val, "symbol", None)
        if isinstance(sym, str):
            return sym
    name = atom.__class__.__name__
    if name.startswith("Dynamic"):
        name = name[len("Dynamic"):]
    return name

def _atom_hcount(atom):
    for attr in ("implicit_hydrogens", "hydrogens", "h", "hydrogen_count"):
        val = getattr(atom, attr, None)
        if val is None:
            continue
        if callable(val):
            try:
                val = val()
            except TypeError:
                continue
        if isinstance(val, (int, float)):
            return int(val)
    return 0

def _bond_order(bond):
    for attr in ("order", "p_order"):
        val = getattr(bond, attr, None)
        if isinstance(val, int):
            return val
    return None

def _is_nitrogen(atom):
    num = getattr(atom, "atomic_number", None)
    if num is None:
        num = getattr(atom, "number", None)
    if num == 7:
        return True
    return _atom_symbol(atom) == "N"

def _is_oxygen(atom):
    num = getattr(atom, "atomic_number", None)
    if num is None:
        num = getattr(atom, "number", None)
    if num == 8:
        return True
    return _atom_symbol(atom) == "O"

def _is_carbonyl_carbon(mol, atom_num):
    atom = mol._atoms.get(atom_num)
    if not atom or _atom_symbol(atom) != "C":
        return False
    bonds = getattr(mol, "_bonds", {})
    for nbr, bond in bonds.get(atom_num, {}).items():
        if _bond_order(bond) != 2:
            continue
        nbr_atom = mol._atoms.get(nbr)
        if nbr_atom and _is_oxygen(nbr_atom):
            return True
    return False

def _amide_by_carbonyl(mol):
    bonds = getattr(mol, "_bonds", {})
    mapping = {}
    for c_num, atom in mol._atoms.items():
        if _atom_symbol(atom) != "C":
            continue
        if not _is_carbonyl_carbon(mol, c_num):
            continue
        for nbr, bond in bonds.get(c_num, {}).items():
            if _bond_order(bond) != 1:
                continue
            nbr_atom = mol._atoms.get(nbr)
            if nbr_atom and _is_nitrogen(nbr_atom):
                mapping[c_num] = nbr
                break
    return mapping

def _select_n_attached_carbon(mol, n_num):
    bonds = getattr(mol, "_bonds", {})
    candidates = []
    for nbr, bond in bonds.get(n_num, {}).items():
        nbr_atom = mol._atoms.get(nbr)
        if not nbr_atom or _atom_symbol(nbr_atom) != "C":
            continue
        if _is_carbonyl_carbon(mol, nbr):
            continue
        order = _bond_order(bond)
        if order not in (1, 4):
            continue
        candidates.append((order, nbr))
    if not candidates:
        return None
    aromatic = [nbr for order, nbr in candidates if order == 4]
    if aromatic:
        return min(aromatic)
    return min(nbr for _, nbr in candidates)

def _expand_mapping_with_substructure(prod_mol, cand_mol, mapping, prod_n=None, cand_n=None):
    try:
        mapping_iter = cand_mol.get_mapping(prod_mol)
    except Exception:
        return mapping, 0

    chosen = None
    for cand_to_prod in mapping_iter:
        if cand_n is not None and prod_n is not None:
            if cand_to_prod.get(cand_n) != prod_n:
                continue
        chosen = cand_to_prod
        break

    if chosen is None:
        return mapping, 0

    prod_to_cand = {p: c for c, p in chosen.items()}
    added = 0
    for p, c in prod_to_cand.items():
        if p not in prod_mol._atoms:
            continue
        if p not in mapping:
            added += 1
        mapping[p] = c
    return mapping, added

def _has_dynamic_bond_4_0(cgr):
    for m_bond in cgr._bonds.values():
        for bond in m_bond.values():
            if isinstance(bond, DynamicBond) and bond.order == 4 and bond.p_order in (None, 0):
                return True
    return False

def _amide_nitrogens(mol):
    bonds = getattr(mol, "_bonds", {})
    amide_nums = []
    for num, atom in mol._atoms.items():
        if not _is_nitrogen(atom):
            continue
        for nbr, bond in bonds.get(num, {}).items():
            if _bond_order(bond) != 1:
                continue
            if _is_carbonyl_carbon(mol, nbr):
                amide_nums.append(num)
                break
    return amide_nums

def _amine_nitrogens(mol, amide_nums):
    amine_nums = {}
    for num, atom in mol._atoms.items():
        if not _is_nitrogen(atom) or num in amide_nums:
            continue
        h_count = _atom_hcount(atom)
        if h_count <= 0:
            continue
        amine_nums[num] = h_count
    return amine_nums

def _prep_molecules(molecules):
    for mol in molecules:
        mol.kekule()
        mol.implicify_hydrogens()
        mol.thiele()

def _find_transamidation_swap(reaction, *, expand=False):
    _prep_molecules(list(reaction.reactants) + list(reaction.products))
    react_amide_by_c = {}
    react_c_idx = {}
    react_n_info = {}
    for idx, mol in enumerate(reaction.reactants):
        amide_map = _amide_by_carbonyl(mol)
        for c_num, n_num in amide_map.items():
            react_amide_by_c[c_num] = n_num
            react_c_idx[c_num] = idx
        for num, atom in mol._atoms.items():
            if not _is_nitrogen(atom):
                continue
            react_n_info[num] = (idx, _atom_hcount(atom), mol)

    prod_amide_by_c = {}
    prod_n_all = set()
    prod_n_mol = {}
    for mol in reaction.products:
        prod_amide_by_c.update(_amide_by_carbonyl(mol))
        for num, atom in mol._atoms.items():
            if _is_nitrogen(atom):
                prod_n_all.add(num)
                prod_n_mol.setdefault(num, mol)

    for c_num, react_n in react_amide_by_c.items():
        prod_n = prod_amide_by_c.get(c_num)
        if prod_n is None:
            continue
        if prod_n != react_n:
            continue
        acyl_idx = react_c_idx.get(c_num)
        candidates = [
            n
            for n, (idx, _, _) in react_n_info.items()
            if idx != acyl_idx and n not in prod_n_all and n != react_n
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda n: (react_n_info[n][1], n),
            reverse=True,
        )
        cand_n = candidates[0]
        mapping = {prod_n: cand_n}
        prod_mol = prod_n_mol.get(prod_n)
        if prod_mol:
            prod_c = _select_n_attached_carbon(prod_mol, prod_n)
        else:
            prod_c = None
        cand_mol = react_n_info[cand_n][2]
        cand_c = _select_n_attached_carbon(cand_mol, cand_n)
        if prod_c and cand_c and cand_c not in prod_mol._atoms:
            mapping[prod_c] = cand_c
        if prod_mol and cand_mol:
            expanded, added = _expand_mapping_with_substructure(
                prod_mol,
                cand_mol,
                mapping,
                prod_n=prod_n,
                cand_n=cand_n,
            )
            if expand or added >= max(3, len(cand_mol) // 2):
                mapping = expanded
        return mapping
    return None

def _remap_product_atoms(reaction, mapping):
    if not mapping:
        return False
    for prod in reaction.products:
        keys = set(mapping) & set(prod._atoms)
        if not keys:
            continue
        prod_map = {k: v for k, v in mapping.items() if k in prod._atoms}
        existing = set(prod._atoms)
        for old, new in list(prod_map.items()):
            if new in existing and new not in prod_map:
                prod_map.pop(old)
        if not prod_map:
            continue
        prod.remap(prod_map)
        return True
    return False

def process_single_route(cgr_pathway, check_trans_error=True):
    for i, reaction in enumerate(cgr_pathway):
        if check_trans_error:
            swap_cgr = reaction.compose()
            expand_swap = _has_dynamic_bond_4_0(swap_cgr)
            swap_pair = _find_transamidation_swap(reaction, expand=expand_swap)
            if swap_pair:
                if _remap_product_atoms(reaction, swap_pair):
                    reaction.flush_cache()
        if i == 0:
            cgr = reaction.compose()
            atoms = reaction.products[0]._atoms.keys()
            if reaction.products[0].atoms_count != max(atoms):
                remapper = find_remap(list(atoms))
                temp_num = max(cgr._atoms)+1
                for key, value in remapper.items():
                    save_val = int(value)
                    cgr.remap({value: temp_num, key: value, value: key})
        else:
            curr_product = reaction.products[0]
            curr_product.kekule()
            curr_product.implicify_hydrogens()
            curr_product.thiele()
            
            for reactant in decomposed.reactants:
                reactant.kekule()
                reactant.implicify_hydrogens()
                reactant.thiele()
                try:
                    if len(reactant) == len(curr_product):
                        curr_remap = next(curr_product.get_mapping(reactant))
                        curr_cgr = reaction.compose()
                        max_num = max(cgr._atoms) + 1
                        curr_decomposed = ReactionContainer.from_cgr(curr_cgr)
                        lg_remap = {}
                        for product in curr_decomposed.products:
                            curr_max_num = max(curr_cgr._atoms) + 1
                            if curr_max_num > max_num:
                                max_num = curr_max_num
                            if len(product) == len(curr_product):
                                continue
                            else:
                                for atom_num in product:
                                    lg_remap[atom_num] = max_num
                                    max_num += 1
                        curr_cgr.remap(lg_remap)
                        curr_cgr.remap(curr_remap)
                        cgr = curr_cgr.compose(cgr)
                except:
                    pass
        decomposed = ReactionContainer.from_cgr(cgr)
    target_cgr = [cgr.substructure(c) for c in cgr.connected_components][0]
    return target_cgr


def _is_route_tree(value: Any) -> bool:
   return isinstance(value, dict) and value.get("type") == "mol"


def _route_smiles(record: Any) -> str | None:
   tree = _route_tree(record)
   if isinstance(tree, dict):
       return tree.get("smiles")
   return None


def _route_tree(record: Any) -> dict[str, Any]:
   if _is_route_tree(record):
       return record
   if isinstance(record, dict) and _is_route_tree(record.get("dict")):
       return record["dict"]
   raise TypeError(f"Unsupported route record: {type(record)!r}")


def _route_id(record: Any, fallback: int) -> Any:
   if isinstance(record, dict):
       for key in ("route_id", "id", "index"):
           if key in record:
               return record[key]
   return fallback


def _normalise_route_record(record: Any, fallback_id: int) -> dict[str, Any]:
   tree = _route_tree(record)
   if isinstance(record, dict) and "dict" in record:
       normalised = dict(record)
       normalised.setdefault("route_id", _route_id(record, fallback_id))
       return normalised
   return {"route_id": fallback_id, "dict": tree}


def iter_route_records(route_collections: Any) -> Iterable[dict[str, Any]]:
   """Yield route records as ``{"route_id": ..., "dict": route_tree}``.

   The previous implementation expected AiZynthFinder ``RouteCollection``
   objects.  PaRoutes stores route trees directly as JSON, so this iterator
   accepts a single route tree, a list of route trees, a dict of ``id -> tree``,
   a list of older ``{"dict": tree}`` records, or a list of RouteCollection-like
   objects exposing a ``dicts`` attribute.
   """
   if _is_route_tree(route_collections):
       yield _normalise_route_record(route_collections, 0)
       return

   if isinstance(route_collections, dict):
       if "dict" in route_collections and _is_route_tree(route_collections["dict"]):
           yield _normalise_route_record(route_collections, 0)
           return
       for idx, key in enumerate(route_collections):
           value = route_collections[key]
           if _is_route_tree(value):
               yield {"route_id": key, "dict": value}
           else:
               yield _normalise_route_record(value, key)
       return

   if hasattr(route_collections, "dicts"):
       for idx, record in enumerate(route_collections.dicts):
           yield _normalise_route_record(record, idx)
       return

   if isinstance(route_collections, Iterable) and not isinstance(route_collections, (str, bytes)):
       for idx, item in enumerate(route_collections):
           if _is_route_tree(item) or (isinstance(item, dict) and _is_route_tree(item.get("dict"))):
               yield _normalise_route_record(item, idx)
           elif hasattr(item, "dicts"):
               for sub_idx, record in enumerate(item.dicts):
                   yield _normalise_route_record(record, f"{idx}:{sub_idx}")
           elif isinstance(item, Iterable) and not isinstance(item, (str, bytes, dict)):
               for sub_idx, record in enumerate(iter_route_records(item)):
                   record.setdefault("route_id", f"{idx}:{sub_idx}")
                   yield record
           else:
               yield _normalise_route_record(item, idx)
       return

   raise TypeError(f"Unsupported route collection: {type(route_collections)!r}")


def _reaction_smiles(node: dict[str, Any]) -> str:
   metadata = node.get("metadata") or {}
   return (
       node.get("smiles")
       or metadata.get("smiles")
       or metadata.get("mapped_reaction_smiles")
       or metadata.get("rsmi")
       or ""
   )


def _route_fingerprint(node: dict[str, Any]) -> tuple[Any, ...]:
   node_type = node.get("type")
   children = tuple(_route_fingerprint(child) for child in node.get("children", []) or [])
   if node_type == "reaction":
       return ("reaction", _reaction_smiles(node), children)
   return (node_type, node.get("smiles"), children)


def normalise_route_tree_for_chython(route: dict[str, Any], *, copy_route: bool = True) -> dict[str, Any]:
   """Return a PaRoutes tree whose reaction ``smiles`` fields chython can read."""
   route = copy.deepcopy(route) if copy_route else route

   def visit(node: dict[str, Any]) -> None:
       if node.get("type") == "reaction":
           smiles = _reaction_smiles(node)
           if not smiles:
               raise ValueError("Reaction node has no chython-readable mapped SMILES")
           node["smiles"] = smiles
       for child in node.get("children", []) or []:
           if isinstance(child, dict):
               visit(child)

   visit(route)
   return route


def route_tree_to_reactions_dict(route: dict[str, Any]) -> dict[int, ReactionContainer]:
   """Convert a PaRoutes tree to ``{step_id: ReactionContainer}`` in synthesis order."""
   route = normalise_route_tree_for_chython(route)
   reactions = []

   def visit(node: dict[str, Any]) -> None:
       for child in node.get("children", []) or []:
           if isinstance(child, dict):
               visit(child)
       if node.get("type") == "reaction":
           reactions.append(smiles_chython(node["smiles"]))

   visit(route)
   return {idx: reaction for idx, reaction in enumerate(reactions)}


def route_tree_to_reactions_list(route: dict[str, Any]) -> list[ReactionContainer]:
   """Convert a PaRoutes tree to a synthesis-ordered list of mapped reactions."""
   return [reaction for _, reaction in sorted(route_tree_to_reactions_dict(route).items())]


def filter_unique_routes(route_collections: Any) -> list[dict[str, Any]]:
   """Filter raw PaRoutes JSON or RouteCollection-like data to unique routes."""
   seen_hashes = set()
   unique_records = []

   for fallback_id, record in enumerate(iter_route_records(route_collections)):
       tree = _route_tree(record)
       route_hash = _route_fingerprint(tree)
       if route_hash in seen_hashes:
           continue
       seen_hashes.add(route_hash)
       unique_records.append(_normalise_route_record(record, fallback_id))

   return unique_records


def extract_pathway_aizynthfinder(node, parent_smiles=None):
   """Recursively extracts a pathway from a reaction tree node."""
   pathway = []
   if node.get('type') == 'reaction':
       for child in node.get('children', []):
           if child.get('type') == 'mol' and 'children' in child:
               for sub in child['children']:
                   pathway.extend(extract_pathway_aizynthfinder(sub, child['smiles']))
       reactants = [c['smiles'] for c in node['children'] if c['type']=='mol'][::-1]
       pathway.append([reactants, parent_smiles])
   else:
       for child in node.get('children', []):
           pathway.extend(extract_pathway_aizynthfinder(child, node.get('smiles')))
   return pathway


def extract_one_route_cgr(data, check_trans_error=True, use_mapped_reaction_smiles=True):
   root = _route_tree(data)
   if use_mapped_reaction_smiles:
       reactions = route_tree_to_reactions_list(root)
       return process_single_route(reactions[::-1], check_trans_error=check_trans_error)

   pathway = extract_pathway_aizynthfinder(root)
   cgr_pathway = route_smi_2_cgr(pathway, reverse=True)
   route_cgr = process_single_route(cgr_pathway, check_trans_error=check_trans_error)
   return route_cgr


def extract_all_route_cgrs(
   route_collection,
   check_trans_error=True,
   collect_errors=False,
   progress_interval=0,
   use_mapped_reaction_smiles=True,
):
   route_cgrs_dict = {}
   errors = []
   for i, data in enumerate(iter_route_records(route_collection)):
       route_id = _route_id(data, i)
       try:
           route_cgr = extract_one_route_cgr(
               data,
               check_trans_error=check_trans_error,
               use_mapped_reaction_smiles=use_mapped_reaction_smiles,
           )
       except Exception as exc:
           error = {
               "route_id": route_id,
               "stage": "extract_route_cgr",
               "target_smiles": _route_smiles(data),
               "error_type": type(exc).__name__,
               "error": str(exc),
               "traceback": traceback.format_exc(),
           }
           if collect_errors:
               errors.append(error)
               continue
           raise
       route_cgrs_dict[route_id] = route_cgr
       if progress_interval and (i + 1) % progress_interval == 0:
           print(f"processed {i + 1} routes; cgrs={len(route_cgrs_dict)} errors={len(errors)}", flush=True)
   if collect_errors:
       return route_cgrs_dict, errors
   return route_cgrs_dict
