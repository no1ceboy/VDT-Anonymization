"""Offline administrative-name lookup. Does not infer person/entity links."""
import csv
from functools import lru_cache
from pathlib import Path
import re

DATA = Path(__file__).resolve().parent.parent/'resources'/'dvhcvn'/'units.tsv'
UNITS = ('thành phố','thị trấn','thị xã','phường','huyện','quận','tỉnh','xã')
UNIT_RE = re.compile(r'(?<!\w)('+ '|'.join(UNITS) +r')[ \t]+',re.I)
ALIAS_RE = re.compile(r'[A-ZĐ](?:[1-9]\d*)?(?!\w)')


@lru_cache(maxsize=1)
def records():
    with DATA.open(encoding='utf-8',newline='') as handle:
        return tuple(csv.DictReader(handle,delimiter='\t'))


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
