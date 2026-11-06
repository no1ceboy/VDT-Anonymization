"""Offline administrative-name lookup. Does not infer person/entity links."""
import csv
import unicodedata
from functools import lru_cache
from pathlib import Path
import re

DATA = Path(__file__).resolve().parents[1] / 'resources' / 'dvhcvn' / 'units.tsv'
UNITS = ('thành phố','thị trấn','thị xã','phường','huyện','quận','tỉnh','xã')
UNIT_RE = re.compile(r'(?<!\w)('+ '|'.join(UNITS) +r')[ \t]+',re.I)
ALIAS_RE = re.compile(r'[A-ZĐ](?:[1-9]\d*)?(?!\w)')


@lru_cache(maxsize=1)
def records():
    with DATA.open(encoding='utf-8',newline='') as handle:
        return tuple(csv.DictReader(handle,delimiter='\t'))


def _normalize_name(value):
    return re.sub(r'\s+', ' ', str(value or '')).strip().casefold()


@lru_cache(maxsize=1)
def records_by_code():
    return {row['code']: row for row in records()}


@lru_cache(maxsize=None)
def record_for_name(unit, name):
    target = _normalize_name(name)
    exact=[row for row in records()
           if row['unit'] == unit and _normalize_name(row['name']) == target]
    if len(exact)==1:
        return exact[0]
    # Historical Vietnamese text alternates between spellings such as
    # ``Hòa`` and ``Hoà``. Use accent-insensitive matching only as a unique
    # fallback; never resolve an ambiguous place name this way.
    def bare(value):
        return ''.join(char for char in unicodedata.normalize('NFD',value)
                       if unicodedata.category(char) != 'Mn')
    matches=[row for row in records()
             if row['unit'] == unit and bare(_normalize_name(row['name'])) == bare(target)]
    if len(matches)==1:
        return matches[0]
    return None


def ancestor_chain(code):
    """Return a unit and all of its parents, nearest first."""
    by_code = records_by_code()
    result=[]
    current=by_code.get(code)
    while current:
        result.append(current)
        parent=current.get('parent','')
        current=by_code.get(parent) if parent else None
    return tuple(result)


@lru_cache(maxsize=None)
def admin_path_candidates(anchor_codes=(), required_units=()):
    """Return valid province/district/commune paths.

    ``anchor_codes`` are visible administrative parents. ``required_units``
    contains ``(role, units)`` pairs for masked components whose unit form
    must be preserved, such as district -> ``huyện`` and commune -> ``xã``.
    """
    anchors=set(anchor_codes)
    requirements={role:set(units) for role,units in required_units}
    paths=[]
    for leaf in records():
        if leaf['level'] != '3':
            continue
        chain=ancestor_chain(leaf['code'])
        codes={row['code'] for row in chain}
        if not anchors.issubset(codes):
            continue
        by_role={
            'commune':next((row for row in chain if row['level']=='3'),None),
            'district':next((row for row in chain if row['level']=='2'),None),
            'province':next((row for row in chain if row['level']=='1'),None),
        }
        if any(not by_role.get(role) or by_role[role]['unit'] not in units
               for role,units in requirements.items()):
            continue
        paths.append(tuple((role,by_role[role]) for role in ('province','district','commune')))
    return tuple(paths)


@lru_cache(maxsize=None)
def names(unit):
    # Numeric ward names are recognized but not sampled for alphabetic aliases.
    return tuple(sorted({r['name'] for r in records() if r['unit']==unit}))


@lru_cache(maxsize=None)
def trie(unit):
    root={}
    for name in names(unit):
        node=root
        for char in ''.join(name.casefold().split()):
            node=node.setdefault(char,{})
        node.setdefault('',set()).add(name)
    return root


def match_name(text,start,unit):
    """Longest known name; whitespace may differ, punctuation cannot."""
    node=trie(unit); best=None
    for end in range(start,min(len(text),start+100)):
        char=text[end]
        if char in ' \t':
            continue
        node=node.get(char.casefold())
        if node is None:
            break
        if '' in node and len(node[''])==1 and (end+1==len(text) or not (text[end+1].isalnum() or text[end+1]=='_')):
            best=(next(iter(node[''])),end+1)
    return best


def parent_components(tail):
    """Read only an adjacent administrative chain, never arbitrary prose."""
    result=[]; pos=0
    while pos<len(tail):
        sep=re.match(r'[ \t,]*',tail[pos:]); pos+=sep.end()
        prefix=UNIT_RE.match(tail,pos)
        if not prefix:
            break
        unit=prefix.group(1).casefold(); start=prefix.end()
        marker=ALIAS_RE.match(tail,start)
        if marker:
            value,pos=marker.group(),marker.end()
        else:
            known=match_name(tail,start,unit)
            if not known:
                words=list(re.finditer(r'[^\W\d_]+',tail[start:]))
                for count in range(min(5,len(words)),0,-1):
                    stop=words[count-1].end()
                    candidate=tail[start:start+stop]
                    record=record_for_name(unit,candidate)
                    if record:
                        known=(record['name'],start+stop)
                        break
            if not known:
                break
            value,pos=known
        result.append(f'{unit} {value}'.casefold())
    return result


def spacing_repairs(text,start,end):
    """Locate recognized administrative names within a NER span.

    Require an intact unit and do not absorb an isolated uppercase alias.
    """
    result=[]
    for prefix in UNIT_RE.finditer(text,start,end):
        unit=prefix.group(1).casefold()
        if ALIAS_RE.match(text,prefix.end(),end):
            continue
        match=match_name(text,prefix.end(),unit)
        if match and match[1]<=end:
            name,stop=match
            original=text[prefix.end():stop]
            if original!=name and ''.join(original.casefold().split())==''.join(name.casefold().split()):
                result.append(dict(start=prefix.end(),end=stop,original=original,normalized=name))
    return result
