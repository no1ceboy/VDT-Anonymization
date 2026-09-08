"""Regression checks for distinct publication conventions (no API required)."""
import unittest
from unittest.mock import patch

from src.build_synthetic_unanonymized import link_document, process_row, marker_value
from src.legal_linking import GeminiResolver


def prediction(text, surfaces):
    entities = []
    for surface, label in surfaces:
        start = text.index(surface)
        entities.append(dict(text=surface, start=start, end=start + len(surface), label=label, score=1))
    return {"entities": entities}


class CourtConventions(unittest.TestCase):
    def test_numbered_codes_and_roundtrip(self):
        text = 'NLQ 1, sinh năm 1953; NLQ2, sinh năm 1983. NLQ1 trình bày. NLC12 làm chứng.'
        _, row, audit, _ = process_row(0, {"id": "test", "markdown": text}, None, 'markdown')
        codes = {e['marker']: e for e in audit['entities'] if e.get('encoding_scheme') == 'procedural_code'}
        self.assertEqual(set(codes), {'NLQ1', 'NLQ2', 'NLC12'})
        self.assertEqual(len(codes['NLQ1']['mentions']), 2)
        self.assertNotEqual(codes['NLQ1']['synthetic_value'], codes['NLQ2']['synthetic_value'])
        self.assertTrue(row['reconstruction_stats']['roundtrip_verified'])
        self.assertTrue(all('synthetic_start' in r and 'original' in r for r in audit['replacements']))

    def test_representative_title_cannot_type_the_principal(self):
        text = 'NLQ1.\nNLQ2. Người đại diện theo pháp luật bà Phạm Thị P, chức vụ: Chi cục trưởng.'
        _, entities, _ = link_document('test', text, None)
        codes = {e['marker']: e for e in entities if e.get('encoding_scheme') == 'procedural_code'}
        self.assertEqual(codes['NLQ1']['label'], 'UNKNOWN')
        self.assertIsNone(codes['NLQ1']['synthetic_value'])
        self.assertEqual(codes['NLQ2']['label'], 'UNKNOWN')
        self.assertIsNone(codes['NLQ2']['synthetic_value'])

    def test_numbered_initials(self):
        text = 'Ông Nguyễn Văn H 12 và ông Nguyễn Văn H13.'
        _, entities, _ = link_document('test', text, prediction(text, [('Nguyễn Văn H 12', 'PER'), ('Nguyễn Văn H13', 'PER')]))
        codes = {e['marker'] for e in entities if e['reconstructable']}
        self.assertEqual(codes, {'H12', 'H13'})
        self.assertIsNone(marker_value('A01'))

    def test_declared_arbitrary_alias(self):
        text = '- Bị đơn: Ông A.\nÔng A trình bày.'
        _, entities, _ = link_document('test', text, None)
        people = [e for e in entities if e['label'] == 'PER' and e.get('marker') == 'A']
        self.assertEqual(len(people), 1)
        self.assertEqual(people[0]['link_status'], 'declared_alias')
        self.assertIsNotNone(people[0]['synthetic_value'])

    def test_same_initial_needs_evidence(self):
        text = 'Ông Trần Hữu T ký. Ông Nguyễn Hùng T xác nhận. Sau đó ông T trình bày.'
        ner = prediction(text, [('Trần Hữu T','PER'), ('Nguyễn Hùng T','PER')])
        _, entities, _ = link_document('test', text, ner)
        self.assertTrue(any(e.get('link_status') == 'ambiguous_marker' for e in entities))
        resolver = GeminiResolver('test', 'unused')
        with patch.object(resolver, 'choose', return_value={'decision':'C0','evidence':'invented evidence'}):
            _, entities, _ = link_document('test', text, ner, resolver)
        self.assertTrue(any(e.get('link_status') == 'ambiguous_marker' for e in entities))
        with patch.object(resolver, 'choose', return_value={'decision':'C0','evidence':'Ông Nguyễn Hùng T xác nhận.', 'reason':'Test of accepted response plumbing, not a semantic accuracy test.'}):
            mentions, _, _ = link_document('test', text, ner, resolver)
        self.assertTrue(any(m.get('link_status') == 'linked_by_llm' for m in mentions))

    def test_compact_company(self):
        text = 'Công tyN giao hồ sơ.'
        _, row, audit, _ = process_row(0, {'markdown':text}, prediction(text,[('Công tyN','PER')]), 'markdown')
        self.assertTrue(any(e['label'] == 'ORG' and e['reconstructable'] for e in audit['entities']))
        self.assertTrue(row['synthetic_markdown'].startswith('Công ty '))

    def test_api_budget(self):
        resolver = GeminiResolver('test', 'unused', max_calls=0)
        self.assertEqual(resolver.choose({})['status'], 'call_budget_exhausted')

    def test_nonentities_never_inherit_a_label(self):
        text = 'V/v giải quyết. Loại A50. Ông Nguyễn Văn V đến xã V, huyện B, tỉnh C.'
        _,row,audit,_=process_row(0,{'markdown':text},prediction(text,[('Nguyễn Văn V','PER')]),'markdown')
        self.assertTrue(row['synthetic_markdown'].startswith('V/v giải quyết. Loại A50.'))
        self.assertEqual(len(audit['rejected_candidates']),2)

    def test_nearby_place_cannot_override_person_ner(self):
        text = 'Ông Nguyễn Văn M. Đến Hà Nội gặp M tại quán.'
        ner=prediction(text,[('Nguyễn Văn M','PER')])
        pos=text.index('M tại')
        ner['entities'].append(dict(text='M',start=pos,end=pos+1,label='PER',score=1.0))
        mentions,entities,_=link_document('test',text,ner)
        reference=next(m for m in mentions if m['start']==pos)
        self.assertEqual(reference['label'],'PER')
        self.assertEqual(reference['ner_evidence'][0]['label'],'PER')
        self.assertIsNone(reference['score'])

    def test_unknown_occurrence_does_not_inherit_person_type(self):
        text='Ông Nguyễn Văn M. Ký hiệu chưa rõ: M'
        _,_,audit,_=process_row(0,{'markdown':text},None,'markdown')
        self.assertFalse(any(p['start']==len(text)-1 for p in audit['replacements']))

    def test_atomic_dotted_initials(self):
        text='Nguyên đơn: Ông N.H.H. Ông N.H.H trình bày.'
        _,row,audit,_=process_row(0,{'markdown':text},None,'markdown')
        replacements=audit['replacements']
        self.assertEqual(len(replacements),2)
        self.assertTrue(all(p['original']=='N.H.H' for p in replacements))
        self.assertEqual(replacements[0]['replacement'],replacements[1]['replacement'])
        self.assertNotIn('.H.H',row['synthetic_markdown'])

    def test_address_units_and_parent_scope(self):
        text='làng A, xã B, huyện C, tỉnh D; làng A, xã E, huyện F, tỉnh G.'
        _,row,audit,_=process_row(0,{'markdown':text},None,'markdown')
        villages=[e for e in audit['entities'] if e['role']=='hamlet']
        self.assertEqual(len(villages),2)
        self.assertTrue(all(p['original'] not in {'làng A','xã B'} for p in audit['replacements']))
        self.assertEqual(row['synthetic_markdown'].count('làng '),2)
        self.assertNotEqual(villages[0]['synthetic_value'],villages[1]['synthetic_value'])

    def test_compact_person_separator(self):
        text='Bị đơn: Bà N. Theo bàN trình bày.'
        _,row,_,_=process_row(0,{'markdown':text},None,'markdown')
        self.assertIn('Theo bà ',row['synthetic_markdown'])

    def test_roman_units_are_reviewed(self):
        text='Địa chỉ: Ấp I, xã B, huyện C, tỉnh D.'
        _,row,_,_=process_row(0,{'markdown':text},None,'markdown')
        self.assertIn('Ấp I',row['synthetic_markdown'])
        self.assertIn('numeral_or_alias',row['review_reasons'])

    def test_unknown_organization_form_is_not_a_bare_company_suffix(self):
        text='NLQ1 ra quyết định.'
        resolver=GeminiResolver('test','unused')
        with patch.object(resolver,'choose',return_value={'decision':'ORG','evidence':text,'reason':'Mocked'}):
            _,row,_,_=process_row(0,{'markdown':text},None,'markdown',resolver)
        self.assertEqual(row['synthetic_markdown'],text)
        self.assertIn('organization_code_needs_full_form',row['review_reasons'])

    def test_unicode_headers_are_not_marker_fragments(self):
        text='NHÂN DANH NƯỚC CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM'
        _,row,audit,_=process_row(0,{'markdown':text},{'entities':[]},'markdown')
        self.assertEqual(row['synthetic_markdown'],text)
        self.assertEqual(audit['entities'],[])

    def test_llm_can_confirm_unknown_reference_but_cannot_hide_review(self):
        text='Ông Nguyễn Văn M. Sau đó M trình bày.'
        resolver=GeminiResolver('test','unused')
        with patch.object(resolver,'choose',return_value={'decision':'C0','evidence':'Sau đó M trình bày.','reason':'Mocked reference decision'}):
            _,row,audit,_=process_row(0,{'markdown':text},{'entities':[]},'markdown',resolver)
        self.assertTrue(any(m.get('link_status')=='linked_by_llm' for e in audit['entities'] for m in e['mentions']))
        self.assertEqual(row['review_status'],'needs_review')

    def test_viewer_checks_source_and_highlights_both_sides(self):
        from src.inspect_entity_links import build_html
        row={'markdown':'Bị đơn: Bà A. Bà A xác nhận.'}
        _,out,audit,_=process_row(0,row,{'entities':[]},'markdown')
        rendered=build_html('0',row,audit,out)
        self.assertIn("class='documents'",rendered)
        self.assertIn('data-entity-id=',rendered)
        with self.assertRaises(ValueError):build_html('0',{'markdown':'changed'},audit,out)
        mismatched=dict(out,synthetic_markdown='Different run')
        with self.assertRaises(ValueError):build_html('0',row,audit,mismatched)

    def test_unmasked_alias_requires_matching_initial(self):
        text='Ông Nguyễn Văn H. Ông Nguyễn Văn Bình ký.'
        _,row,_,_=process_row(0,{'markdown':text},prediction(text,[('Nguyễn Văn H','PER'),('Nguyễn Văn Bình','PER')]),'markdown')
        self.assertNotIn('possible_unmasked_alias',row['review_reasons'])
        text='Ông Nguyễn Văn H. Ông Nguyễn Văn Hùng ký.'
        _,row,_,_=process_row(0,{'markdown':text},prediction(text,[('Nguyễn Văn H','PER'),('Nguyễn Văn Hùng','PER')]),'markdown')
        self.assertIn('possible_unmasked_alias',row['review_reasons'])

    def test_common_vietnamese_family_prefix_can_generate_masked_given_name(self):
        text='Bị đơn: Ông Nguyễn Phát Q, sinh năm 1979. Ông Q có mặt.'
        _,row,_,details=process_row(
            0, {'markdown':text},
            prediction(text,[('Nguyễn Phát Q','PER'),('Q','PER')]),
            'markdown')
        person=next(entity for entity in details['entities'] if entity['label']=='PER')
        self.assertTrue(person['reconstructable'])
        self.assertTrue(person['synthetic_value'].startswith('Nguyễn Phát '))
        self.assertEqual(person['synthetic_value'].split()[-1][0], 'Q')
        self.assertNotIn('unsupported_name_prefix', person['review_reasons'])

    def test_gender_conflict_does_not_block_valid_initial(self):
        text='Ông Nguyễn Phát Q. Bà Nguyễn Phát Q.'
        _,row,_,details=process_row(
            0, {'markdown':text},
            prediction(text,[('Nguyễn Phát Q','PER')]),
            'markdown')
        person=next(entity for entity in details['entities'] if entity['label']=='PER')
        self.assertTrue(person['reconstructable'])
        self.assertEqual(person['synthetic_value'].split()[-1][0], 'Q')

    def test_address_numbers_use_small_role_ranges(self):
        text='Địa chỉ: Số X, tầng Y, phòng Z, ấp A, xã B, huyện C.'
        _,_,_,details=process_row(
            0, {'markdown':text},
            prediction(text,[('X','LOC'),('Y','LOC'),('Z','LOC'),('A','LOC'),('B','LOC'),('C','LOC')]),
            'markdown')
        values={e['role']:int(e['synthetic_value']) for e in details['entities'] if e['label']=='ADDR'}
        self.assertLessEqual(values['floor'],8)
        self.assertGreaterEqual(values['floor'],1)
        self.assertLessEqual(values['house_number'],300)
        self.assertLessEqual(values['room'],508)

    def test_bank_legal_form_is_not_section_number(self):
        text='Nguyên đơn Ngân hàng thương mại cổ phần X'
        _,row,_,details=process_row(
            0, {'markdown':text},
            prediction(text,[('X','ORG')]),
            'markdown')
        organization=next(entity for entity in details['entities'] if entity['label']=='ORG')
        self.assertTrue(organization['reconstructable'])
        self.assertTrue(organization['synthetic_value'])
        self.assertNotEqual(organization['synthetic_value'], 'X')
        self.assertEqual(details['replacements'][0]['original'], 'X')

    def test_unknown_type_llm_propagates_code(self):
        resolver = GeminiResolver('test', 'unused')
        text = 'NLQ1 nhận thông báo. NLQ 1 trả lời.'
        with patch.object(resolver, 'choose', return_value={
                'decision':'ORG','evidence':'NLQ1 nhận thông báo.', 'reason':'Mocked type decision'}):
            mentions, entities, _ = link_document('test',text,None,resolver)
        code = [e for e in entities if e.get('marker')=='NLQ1'][0]
        self.assertEqual(code['label'],'ORG')
        self.assertEqual(len(code['mentions']),2)

    def test_transport_payload_and_error(self):
        import io,json,urllib.error
        resolver=GeminiResolver('model','secret')
        response={'candidates':[{'content':{'parts':[{'text':json.dumps({'decision':'ABSTAIN','evidence':'','reason':'unclear'})}]}}]}
        with patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(response).encode())) as call:
            self.assertEqual(resolver.choose({})['decision'],'ABSTAIN')
            request=call.call_args.args[0]
            self.assertNotIn('secret',request.full_url)
            self.assertEqual(json.loads(request.data)['generationConfig']['responseMimeType'],'application/json')
        with patch('urllib.request.urlopen',side_effect=urllib.error.HTTPError('url',429,'quota',{},None)):
            self.assertEqual(resolver.choose({})['status'],'http_429')


if __name__ == '__main__':
    unittest.main()
