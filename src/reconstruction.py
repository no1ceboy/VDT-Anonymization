"""Evidence-first reconstruction: observe -> type -> link -> plan -> validate.

Each stage consumes the previous stage's decisions. Lexical similarity alone
cannot establish a type. Uncertain observations never propagate labels.
"""
import hashlib
import re
from collections import defaultdict

try:
    from .synthetic_lexicon import FAMILY_NAMES, GIVEN_NAMES, NAME_PREFIX_TOKENS, LOCATION_NAMES, SYNTHETIC_ORGANIZATION_NAMES
except ImportError:
    from synthetic_lexicon import FAMILY_NAMES, GIVEN_NAMES, NAME_PREFIX_TOKENS, LOCATION_NAMES, SYNTHETIC_ORGANIZATION_NAMES

VERSION = "evidence-v2"
CODE_RE = re.compile(r"(?:NLQ|NLC)\s*\d+")
DOTTED = r"(?:[A-ZĐ]\.){1,5}[A-ZĐ](?:[1-9]\d*)?"
TOKEN_RE = re.compile(
    rf"(?<![A-ZĐ\d_])(?:NLQ\s*\d+|NLC\s*\d+|{DOTTED}|"
    r"[IVX]{2,6}|(?:Th|Ph|Tr|Ng|Ch|Kh|Nh)|[A-ZĐ](?:[ \t]*[1-9]\d*)?)(?!\w)"
)
PERSON_TITLE = re.compile(r"(?<!\w)(ông|bà|anh|chị|cháu|cô|chú|bác|em)\s*$", re.I)
DECLARATION = re.compile(r"(nguyên đơn|bị đơn|bị cáo|bị hại|người làm chứng|người khởi kiện|người bị kiện)\s*:\s*(?:(?:ông|bà|anh|chị|cháu)\s*)?$", re.I)
ORG_PREFIX = re.compile(r"(?<!\w)(ngân hàng(?: thương mại cổ phần)?|công ty(?: trách nhiệm hữu hạn| cổ phần| tnhh)?|hợp tác xã|chi cục(?: thi hành án dân sự)?|ủy ban nhân dân)\s*$", re.I)
ADDRESS_UNITS = {
    "số": "house_number", "số nhà": "house_number", "tầng": "floor", "lầu": "floor",
    "phòng": "room", "căn": "room", "thửa": "parcel", "lô": "parcel",
    "ấp": "hamlet", "thôn": "hamlet", "làng": "hamlet", "khóm": "hamlet",
    "khu phố": "hamlet", "tổ": "hamlet", "tổ dân phố": "hamlet",
    "xã": "commune", "phường": "commune", "thị trấn": "commune",
    "huyện": "district", "quận": "district", "thị xã": "district",
    "thành phố": "city", "tỉnh": "province", "đường": "street", "lộ": "street",
    "quốc lộ": "road_number", "ngõ": "street", "hẻm": "street",
}
UNIT_PATTERN = "|".join(re.escape(s) for s in sorted(ADDRESS_UNITS, key=len, reverse=True))
ADDRESS_PREFIX = re.compile(rf"(?<!\w)({UNIT_PATTERN})\s*$", re.I)
# Capture only the administrative unit and its immediate value. The previous
# expression consumed surrounding legal prose, causing the same `huyện V` to
# receive different scopes at different positions in a document.
PARENT_RE = re.compile(r"(?:xã|phường|thị trấn|huyện|quận|thị xã|thành phố|tỉnh)\s+[^\s,;:.()\-]+", re.I)
NONENTITY_PREFIX = re.compile(r"(?:loại|mã|ký hiệu|biển số|số hiệu|điểm|hạng|nhóm)\s*$", re.I)
FOREIGN_RE = re.compile(r"Đài Loan|Trung Quốc|Australia|Sydney|Hoa Kỳ|Hàn Quốc|Nhật Bản", re.I)
GENDER = {"ông":"male", "anh":"male", "chú":"male", "bà":"female", "chị":"female", "cô":"female"}
NEUTRAL = ["Anh", "An", "Bình", "Hà", "Minh", "Thanh", "Tâm"]
MIDDLES = ["Văn", "Thị", "Hữu", "Đức", "Ngọc", "Quốc", "Hoàng", "Thanh", "Minh", "Gia", "Xuân"]

# Broader, familiar Vietnamese place vocabulary. These are synthetic values,
# not attempts to recover the source location. Keeping several alternatives
# per administrative role prevents every document from cycling through the
# same small set of two-syllable names.
PROVINCE_NAMES = [
    "An Giang", "Bắc Giang", "Bắc Ninh", "Bình Định", "Bình Dương",
    "Bình Phước", "Cà Mau", "Cao Bằng", "Đà Nẵng", "Đắk Lắk",
    "Điện Biên", "Đồng Nai", "Đồng Tháp", "Gia Lai", "Hà Nam",
    "Hà Tĩnh", "Hải Dương", "Hậu Giang", "Hòa Bình", "Khánh Hòa",
    "Kiên Giang", "Lâm Đồng", "Lạng Sơn", "Lào Cai", "Long An",
    "Nam Định", "Nghệ An", "Ninh Bình", "Ninh Thuận", "Phú Thọ",
    "Phú Yên", "Quảng Bình", "Quảng Nam", "Quảng Ngãi", "Quảng Ninh",
    "Sóc Trăng", "Sơn La", "Tây Ninh", "Thái Bình", "Thái Nguyên",
    "Thanh Hóa", "Tiền Giang", "Trà Vinh", "Tuyên Quang", "Vĩnh Long",
    "Vĩnh Phúc", "Yên Bái",
]

LOCATION_VARIANTS = {
    "district": [
        "An Bình", "Bình Minh", "Cao Lãnh", "Đức Hòa", "Đông Hòa",
        "Hòa Thành", "Long Thành", "Minh Long", "Nam Giang", "Phú Bình",
        "Quế Sơn", "Tân Châu", "Thuận Thành", "Vạn Ninh", "Xuân Lộc",
    ],
    "commune": [
        "An Hòa", "Bình An", "Cẩm Tú", "Đông Phú", "Hòa Bình",
        "Long Hưng", "Minh Tân", "Nam Sơn", "Phú An", "Quang Trung",
        "Tân Lập", "Thanh Sơn", "Vĩnh Hòa", "Xuân Thịnh", "Yên Phú",
    ],
    "hamlet": [
        "An Bình", "Bình Hòa", "Cầu Mới", "Đông Bình", "Hòa Phú",
        "Long Bình", "Minh Tân", "Nam Bình", "Phú Bình", "Quang Trung",
        "Tân Bình", "Thanh Bình", "Vĩnh Bình", "Xuân Bình", "Yên Bình",
    ],
}


def normalize(value):
    return re.sub(r"\s+", " ", value).strip().casefold()


def marker_value(value):
    value = re.sub(r"\s+", "", str(value or ""))
    return value if re.fullmatch(rf"(?:NLQ|NLC)\d+|{DOTTED}|[A-ZĐ](?:[1-9]\d*)?|Th|Ph|Tr|Ng|Ch|Kh|Nh", value) else None


def stable_slot(doc_id, key):
    return int.from_bytes(hashlib.sha256(f"{doc_id}|{key}".encode()).digest()[:8], 'big')


def issue(mention, reason):
    if reason not in mention["review_reasons"]:
        mention["review_reasons"].append(reason)


def prefix_before(text, start):
    """Recognize a contiguous name prefix; never scan across punctuation."""
    before = text[max(0,start-100):start]
    words = list(re.finditer(r"[^\W\d_]+", before))
    chosen = []
    end = len(before)
    for word in reversed(words):
        if before[word.end():end].strip():
            break
        token = word.group()
        if not token[0].isupper() or normalize(token) not in NAME_PREFIX_TOKENS:
            break
        chosen.insert(0, token)
        end = word.start()
        if len(chosen) == 5:
            break
    if not chosen or normalize(chosen[0]) not in {normalize(s) for s in FAMILY_NAMES}:
        return start, ""
    return max(0,start-100)+end, " ".join(chosen)


def observations(text, ner_record):
    """Keep original NER evidence separate from final entity decisions."""
    result = []
    for entity in (ner_record or {}).get('entities', []):
        start, end = entity.get('start'), entity.get('end')
        if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(text)):
            continue
        label = str(entity.get('label','')).upper().split('-')[-1]
        label = {'PERSON':'PER', 'LOCATION':'LOC', 'ORGANIZATION':'ORG'}.get(label,label)
        if label in {'PER','LOC','ORG'}:
            result.append({'start':start,'end':end,'label':label,'score':entity.get('score'),'text':text[start:end]})
    return result


def detect_candidates(text, ner_record):
    ner = observations(text, ner_record)
    candidates = []
    for match in TOKEN_RE.finditer(text):
        start, end = match.span()
        token = re.sub(r"\s+", "", match.group())
        before = text[max(0,start-150):start]
        after = text[end:end+120]
        overlapping = [n for n in ner if n['start'] <= start and n['end'] >= end]
        m = dict(start=start,end=end,text=text[start:end],marker=token,label='UNKNOWN',
                 score=None,ner_evidence=overlapping,detectors=['lexical'],
                 link_evidence='unconfirmed',encoding_scheme='unconfirmed',link_status='unconfirmed_marker',
                 review_reasons=[],person_anchor=None,role='unknown')
        candidates.append(m)
        # Reject lexical fragments of identifiers. Rejection is occurrence-local.
        if start and text[start-1].isalnum() and not text[start-1].islower():
            m['link_status']='rejected_nonentity'; m['link_evidence']='inside_uppercase_word_or_identifier'; continue
        if re.match(r'[Vv]/[Vv](?!\w)',text[start:start+4]) or (start and text[start-1] in '/_'):
            m['link_status']='rejected_nonentity'; m['link_evidence']='abbreviation_or_identifier'; continue
        if NONENTITY_PREFIX.search(before):
            m['link_status']='rejected_nonentity'; m['link_evidence']='technical_or_enumeration_context'; continue
        # Section-number protection is intentionally case-sensitive. A legal
        # organization such as ``Ngân hàng thương mại cổ phần X`` contains
        # lowercase ``phần`` immediately before its anonymized marker; that
        # must not be mistaken for a ``PHẦN X`` heading.
        if re.search(r'(?:Page|PHẦN|MỤC)\s*$',before):
            m['link_status']='rejected_nonentity'; m['link_evidence']='section_number'; continue

        address = ADDRESS_PREFIX.search(before)
        organization = ORG_PREFIX.search(before)
        title = PERSON_TITLE.search(before)
        declaration = DECLARATION.search(before)
        prefix_start, prefix = prefix_before(text,start)
        if CODE_RE.fullmatch(match.group()):
            m.update(encoding_scheme='procedural_code',person_anchor='code:'+token,
                     link_status='unresolved_type',link_evidence='explicit_procedural_code',
                     procedural_role='witness' if token.startswith('NLC') else 'related_party')
            if title or re.match(r'\s*[,;]?\s*sinh năm\s*:?\s*\d{4}',after,re.I) or token.startswith('NLC'):
                m['label']='PER'
            if organization:
                m['label']='ORG'; m['organization_form']=organization.group(1)
        elif address:
            unit=normalize(address.group(1)); role=ADDRESS_UNITS[unit]
            m.update(label='ADDR' if role in {'house_number','floor','room','parcel','road_number'} else 'LOC',
                     role=role,address_unit=unit,encoding_scheme='component_alias',link_status='typed',
                     link_evidence='direct_address_grammar')
            # An actual numbered unit and a publication alias are indistinguishable
            # from typography alone. Never convert Roman I into a random place.
            if re.fullmatch(r'I+|IV|VI+|IX|XI+', token):
                issue(m,'numeral_or_alias')
            tail = re.split(r'[;\n.!?]',text[end:end+250])[0]
            parents = [normalize(p.group()) for p in PARENT_RE.finditer(tail)]
            m['address_parents']=parents
            m['identity_scope']='|'.join(parents)
            block = text[max(0,start-120):end+250].split('\nNỘI DUNG')[0]
            if FOREIGN_RE.search(block): issue(m,'unsupported_foreign_address')
            if role=='province': issue(m,'province_alias_requires_review')
            if role=='road_number': issue(m,'road_number_or_alias')
        elif organization:
            m.update(label='ORG',role='organization',encoding_scheme='component_alias',
                     organization_form=normalize(organization.group(1)),link_status='typed',link_evidence='direct_organization_grammar')
        elif prefix:
            m.update(label='PER',role='person',start=prefix_start,text=text[prefix_start:end],
                     name_prefix=prefix,encoding_scheme='numbered_initial' if any(c.isdigit() for c in token) else 'name_initial',
                     person_anchor=normalize(prefix),link_status='explicit_anchor',link_evidence='contiguous_name_prefix')
            title=PERSON_TITLE.search(text[max(0,prefix_start-80):prefix_start])
            declaration=DECLARATION.search(text[max(0,prefix_start-150):prefix_start])
        elif title or declaration:
            m.update(label='PER',role='person',encoding_scheme='dotted_initials' if '.' in token else 'whole_name_alias',
                     link_status='typed',link_evidence='direct_person_grammar')
        elif overlapping:
            labels={n['label'] for n in overlapping}
            m.update(detectors=['lexical','ner'],encoding_scheme='dotted_initials' if '.' in token else 'whole_name_alias')
            if len(labels)==1:
                m.update(label=next(iter(labels)),link_status='typed',link_evidence='ner_observation')
            else:
                m.update(link_status='type_conflict',type_candidates=sorted(labels));issue(m,'conflicting_ner_types')
            # Retain a complete NER name prefix, but never invent a Vietnamese
            # replacement for an unsupported name family (e.g. Liao, Chih).
            if m['label']=='PER':
                best=max(overlapping,key=lambda n:n['end']-n['start'])
                surface=text[best['start']:start].strip()
                surface=re.sub(r'^(?:ông|bà|anh|chị|cháu)\s+','',surface,flags=re.I)
                if surface:
                    m.update(start=best['start'],text=text[best['start']:end],name_prefix=surface,
                             person_anchor=normalize(surface),encoding_scheme='name_initial')
                    # A partially masked NER span is not automatically an
                    # unsupported name.  Recognized Vietnamese family names
                    # (e.g. Nguyễn Phát Q) are valid anchors and should be
                    # allowed to reach the compatible given-name generator.
                    family_head=normalize(surface.split()[0]) if surface.split() else ''
                    known_families={normalize(name) for name in FAMILY_NAMES}
                    if family_head not in known_families:
                        issue(m,'unsupported_name_prefix')
        # All other tokens stay UNKNOWN. Knowing that another M is a person
        # cannot turn this occurrence into a person or a place.
        if title:
            m['gender']=GENDER.get(title.group(1).casefold())
        if m.get('name_prefix'):
            if re.search(r'\bThị\b',m['name_prefix']):m['gender']='female'
            elif re.search(r'\bVăn\b',m['name_prefix']):m['gender']='male'
        if declaration:
            m['declaration']=True; m['procedural_role']=normalize(declaration.group(1))
        # A lexical fragment must not be reconstructed as an independent person.
        next_word=re.match(r'\s+([^\W\d_]+)',after)
        if next_word and normalize(token+next_word.group(1)) in {normalize(s) for s in FAMILY_NAMES}:
            issue(m,'fragmented_surname')
        if m.get('name_prefix') and re.match(r'\s+[a-zđ](?!\w)',after):
            issue(m,'fragmented_initial')
        if '.' in token and m['label']=='PER':
            m.update(encoding_scheme='dotted_initials',person_anchor='dotted:'+token,link_status='explicit_code')
        if m['label']=='LOC' and not address:
            issue(m,'location_without_address_structure')
        if m['label']=='ORG' and not organization and m['encoding_scheme']!='procedural_code':
            issue(m,'organization_without_form')
        if overlapping:
            m['detectors']=sorted(set(m['detectors'])|{'ner'})
            m['score']=None # NER confidence is retained only in ner_evidence.
        if m['link_status']=='unconfirmed_marker':issue(m,'unconfirmed_marker')
    return candidates, ner


def link_candidates(text, mentions, resolver=None, doc_id=''):
    """Build typed registries, then attach independently typed references."""
    registry=defaultdict(lambda:defaultdict(list))
    codes=defaultdict(list)
    for m in mentions:
        if m['encoding_scheme']=='procedural_code': codes[m['marker']].append(m)
        elif m['label']=='PER' and m.get('person_anchor'):
            registry[m['marker']][m['person_anchor']].append(m)
    for code, peers in codes.items():
        types={m['label'] for m in peers if m['label']!='UNKNOWN'}
        for m in peers:
            if len(types)==1:
                m.update(label=next(iter(types)),link_status='explicit_code',role='person' if 'PER' in types else 'organization')
            else:
                m.update(label='UNKNOWN',link_status='unresolved_type')
                issue(m,'unknown_participant_type' if not types else 'conflicting_participant_types')
    if resolver:
        resolver.resolve_types(doc_id,text,mentions)
    declarations=defaultdict(list)
    for m in mentions:
        if m['label']=='PER' and m.get('declaration') and not m.get('person_anchor'):
            declarations[m['marker']].append(m)
    for code, peers in declarations.items():
        if len(peers)==1 and not registry[code]:
            registry[code]['alias:'+code]=peers
    for m in mentions:
        if m['label']!='PER' or m.get('person_anchor'):continue
        choices=registry[m['marker']]
        if len(choices)==1:
            anchor=next(iter(choices)); m.update(person_anchor=anchor,link_status='linked_by_unique_anchor')
            if anchor.startswith('alias:'):m['link_status']='declared_alias'
        else:
            m['link_status']='ambiguous_marker';issue(m,'ambiguous_identity' if choices else 'missing_identity_anchor')
    if resolver:
        resolver.resolve(doc_id,text,mentions)
        for m in mentions:
            if m.get('llm_decision',{}).get('status')=='accepted':
                m['review_reasons']=[r for r in m['review_reasons'] if r not in {'ambiguous_identity','missing_identity_anchor','unknown_participant_type','conflicting_ner_types','unconfirmed_marker'}]
                issue(m,'llm_decision_requires_review')
    # Unmasked NER names are evidence for review, not reliable identities to
    # copy into the synthetic dataset. No implicit "known_full_name" recovery.
    return mentions


def person_value(doc_id,key,group,used):
    genders={m.get('gender') for m in group if m.get('gender')}
    # Gender improves realism when available, but it is not an identity
    # constraint. Court text can omit titles or contain inconsistent ones.
    # Fall back to a neutral pool instead of blocking a valid initial.
    gender=next(iter(genders)) if len(genders)==1 else None
    prefixes={m.get('name_prefix') for m in group if m.get('name_prefix')}
    if len(prefixes)>1:return None,'conflicting_name_prefixes'
    prefix=next(iter(prefixes)) if prefixes else ''
    marker=group[0]['marker']; dotted=group[0]['encoding_scheme']=='dotted_initials'
    families=FAMILY_NAMES
    middles=['Thị'] if gender=='female' else ['Văn'] if gender=='male' else ['']
    neutral_pool=list(dict.fromkeys(
        n for gender_values in GIVEN_NAMES.values() for values in gender_values.values() for n in values
    ))
    pool=list(dict.fromkeys(n for values in GIVEN_NAMES[gender].values() for n in values)) if gender else neutral_pool
    if dotted:
        letters=marker.split('.')
        if len(letters)!=3 or any(len(s)!=1 for s in letters):return None,'unsupported_dotted_initials'
        families=[n for n in families if n[0]==letters[0]]
        middles=[n for n in MIDDLES if n[0]==letters[1]]
        if gender=='female':middles=[n for n in middles if n not in {'Văn','Hữu','Quốc'}]
        if gender=='male':middles=[n for n in middles if n!='Thị']
        pool=[n for n in pool if n[0]==letters[-1]]
    elif prefix:
        initial=re.sub(r'\d+$','',marker)
        pool=[n for n in pool if n.startswith(initial)]
        if not pool and gender:
            pool=[n for n in neutral_pool if n.startswith(initial)]
    if not pool or not families or not middles:return None,'no_compatible_synthetic_name'
    offset=stable_slot(doc_id,key)
    for i in range(len(pool)*len(families)*len(middles)):
        p=pool[(offset+i)%len(pool)]
        family=families[((offset+i)//len(pool))%len(families)]
        middle=middles[((offset+i)//(len(pool)*len(families)))%len(middles)]
        value=f'{prefix} {p}' if prefix else ' '.join(s for s in (family,middle,p) if s)
        if value not in used:
            used.add(value);return value,None
    return None,'synthetic_name_pool_exhausted'


def replacement_value(doc_id,key,group,used):
    first=group[0]; label=first['label']; role=first['role']
    hard={r for m in group for r in m['review_reasons'] if r!='llm_decision_requires_review'}
    if hard:return None,sorted(hard)[0]
    if label=='PER':return person_value(doc_id,key,group,used)
    if label=='UNKNOWN':return None,'unknown_entity_type'
    slot=stable_slot(doc_id,key)
    if label=='ORG':
        if first['encoding_scheme']=='procedural_code':
            return None,'organization_code_needs_full_form'
        pool=SYNTHETIC_ORGANIZATION_NAMES
    elif label=='LOC':
        if not first.get('address_parents'):return None,'missing_address_parent'
        unit=role if role in LOCATION_NAMES else 'generic'
        pool=list(dict.fromkeys(s for values in LOCATION_NAMES[unit].values() for s in values))
        if unit=='province':
            pool=list(dict.fromkeys(PROVINCE_NAMES+pool))
        else:
            pool=list(dict.fromkeys(LOCATION_VARIANTS.get(unit,[])+pool))
    elif label=='ADDR':
        ranges={
            'floor': (1, 8),
            'house_number': (1, 300),
            'room': (101, 508),
            'parcel': (1, 300),
            'road_number': (1, 300),
        }
        low,high=ranges.get(role,(1,300))
        return str(low+slot%(high-low+1)),None
    else:return None,'unsupported_entity_type'
    for i in range(len(pool)):
        value=pool[(slot+i)%len(pool)]
        if value not in used:used.add(value);return value,None
    return None,'synthetic_pool_exhausted'


def build_entities(doc_id,text,mentions,ner):
    groups=defaultdict(list)
    for m in mentions:
        if m['link_status']=='rejected_nonentity':continue
        if m['encoding_scheme']=='procedural_code':key=('code',m['marker'])
        elif m['label']=='PER' and m.get('person_anchor'):key=('PER',m['marker'],m['person_anchor'])
        elif m['label'] in {'LOC','ADDR'} and m.get('address_unit'):
            key=(m['label'],m['address_unit'],m['marker'],m.get('identity_scope',''))
        elif m['label']=='ORG' and m.get('organization_form'):
            key=('ORG',m['organization_form'],m['marker'])
        else:key=('unresolved',m['start'],m['end'])
        groups[key].append(m)
    entities=[]; planned=[]; used=set()
    # Avoid generating an identity already present in the source.
    used.update(n['text'] for n in ner if n['label']=='PER')
    for index,(key,group) in enumerate(sorted(groups.items(),key=lambda item:min(m['start'] for m in item[1])),1):
        group.sort(key=lambda m:m['start']); first=group[0]
        for n in ner:
            if n['label']!='PER' or any(n['start']<=m['start'] and n['end']>=m['end'] for m in group):continue
            if TOKEN_RE.search(n['text']):continue
            prefix=next((m.get('name_prefix') for m in group if m.get('name_prefix')),None)
            initial = re.sub(r'\d+$','',first['marker'])
            if (prefix and normalize(n['text']).startswith(normalize(prefix)+' ')
                    and n['text'].split()[-1].startswith(initial)):
                for m in group:issue(m,'possible_unmasked_alias')
                break
        value,reason=replacement_value(doc_id,str(key),group,used)
        if reason:
            for m in group:issue(m,reason)
        eid=f"{first['label']}_{index:04d}"
        for m in group:m['entity_id']=eid
        flags=sorted({r for m in group for r in m['review_reasons']})
        entities.append(dict(entity_id=eid,label=first['label'],role=first['role'],marker=first['marker'],
                             person_anchor=first.get('person_anchor'),encoding_scheme=first['encoding_scheme'],
                             procedural_role=first.get('procedural_role'),known_full_name=None,
                             synthetic_value=value,reconstructable=value is not None,
                             link_status=first['link_status'],review_reasons=flags,
                             name_rule='compatible_synthetic' if value and first['label']=='PER' else None,
                             marker_initial_preserved=None,mentions=group))
        if value:
            for m in group:
                replacement=value
                if m['start'] and text[m['start']-1].isalnum():replacement=' '+replacement
                if m['end']<len(text) and text[m['end']].isalnum():replacement+=' '
                planned.append(dict(entity_id=eid,label=m['label'],start=m['start'],end=m['end'],replacement=replacement))
    # Overlaps invalidate every affected entity. Never silently choose one edit.
    conflicted=set()
    ordered=sorted(planned,key=lambda p:p['start'])
    for i,p in enumerate(ordered):
        for q in ordered[i+1:]:
            if q['start']>=p['end']:break
            conflicted.update((p['entity_id'],q['entity_id']))
    for entity in entities:
        if entity['entity_id'] in conflicted:
            entity.update(reconstructable=False,synthetic_value=None)
            entity['review_reasons'].append('overlapping_replacement_spans')
            for m in entity['mentions']:issue(m,'overlapping_replacement_spans')
    return entities,[p for p in planned if p['entity_id'] not in conflicted]


def link_document(doc_id,text,ner_record,resolver=None):
    mentions,ner=detect_candidates(text,ner_record)
    link_candidates(text,mentions,resolver,doc_id)
    entities,planned=build_entities(doc_id,text,mentions,ner)
    return mentions,entities,planned


def apply_replacements(text,replacements):
    ordered=sorted(replacements,key=lambda p:(p['start'],p['end']))
    previous=0
    for p in ordered:
        if not (previous<=p['start']<p['end']<=len(text)):
            raise ValueError('Invalid or overlapping replacement plan')
        previous=p['end']
    result=text
    for p in reversed(ordered):result=result[:p['start']]+p['replacement']+result[p['end']:]
    return result,ordered
