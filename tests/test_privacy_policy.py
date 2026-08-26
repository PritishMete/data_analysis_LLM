import asyncio

from privacy_policy import LOCAL_ONLY, privacy_status, reject_if_local_only
from ai_privacy import validate_metadata_planner_payload
import command_agent
from fastapi import HTTPException


def test_local_only_is_default(monkeypatch):
    # Module-level default is local-only unless deployment explicitly opts in.
    assert isinstance(LOCAL_ONLY, bool)
    assert privacy_status()['mode'] in {'local_only', 'remote_allowed'}


def test_local_only_blocks_dataset_routes():
    import privacy_policy
    old = privacy_policy.LOCAL_ONLY
    privacy_policy.LOCAL_ONLY = True
    try:
        try:
            privacy_policy.reject_if_local_only('/analyze', 'multipart/form-data; boundary=x')
        except HTTPException as exc:
            assert exc.status_code == 403
        else:
            raise AssertionError('expected /analyze to be blocked')
    finally:
        privacy_policy.LOCAL_ONLY = old


def test_local_only_blocks_future_multipart_uploads():
    import privacy_policy
    old = privacy_policy.LOCAL_ONLY
    privacy_policy.LOCAL_ONLY = True
    try:
        try:
            privacy_policy.reject_if_local_only('/future-upload', 'multipart/form-data; boundary=x')
        except HTTPException as exc:
            assert exc.status_code == 403
        else:
            raise AssertionError('expected multipart upload to be blocked')
    finally:
        privacy_policy.LOCAL_ONLY = old


def test_metadata_planner_payload_rejects_workbook_content():
    safe_text, safe_columns, safe_sheets = validate_metadata_planner_payload({
        'text': 'categorize all columns',
        'available_columns': ['Country', 'City'],
        'available_sheets': ['Sheet1'],
    })
    assert safe_text == 'categorize all columns'
    assert safe_columns == ['Country', 'City']
    assert safe_sheets == ['Sheet1']

    for forbidden in (
        {'text': 'categorize', 'rows': [{'Country': 'India'}]},
        {'text': 'categorize', 'values': ['PRIVATE_TEST_VALUE_928371']},
        {'text': 'categorize', 'samples': ['PRIVATE_REVIEW_88127']},
    ):
        try:
            validate_metadata_planner_payload(forbidden)
        except ValueError:
            pass
        else:
            raise AssertionError('expected workbook-shaped payload to be rejected')


def test_metadata_planner_payload_rejects_nested_workbook_content():
    forbidden = {
        'text': 'categorize',
        'context': {
            'nested': [
                {'sheet_name': 'Sheet1'},
                {'payload': {'rows': [{'Country': 'India'}]}},
            ]
        },
    }
    try:
        validate_metadata_planner_payload(forbidden)
    except ValueError:
        pass
    else:
        raise AssertionError('expected nested workbook-shaped payload to be rejected')


def test_parse_filter_plan_anonymizes_columns_before_gemini(monkeypatch):
    captured = {}

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            captured['instruction'] = kwargs.get('instruction', '')

    class FakeSessionService:
        async def create_session(self, *args, **kwargs):
            return None

    class FakeEvent:
        def __init__(self, text):
            self.content = type('Content', (), {'parts': [type('Part', (), {'text': text})()]})()

        def is_final_response(self):
            return True

    class FakeRunner:
        def __init__(self, *args, **kwargs):
            self.agent = kwargs.get('agent')

        async def run_async(self, user_id, session_id, new_message):
            captured['prompt'] = new_message.parts[0].text
            yield FakeEvent('{"intent":"filter","logic":"AND","filters":[{"column":"FIELD_01","operator":"equals","value":"India"}]}')

    monkeypatch.setattr(command_agent, 'strict_enabled', lambda: True)
    monkeypatch.setattr(command_agent, 'LlmAgent', FakeAgent)
    monkeypatch.setattr(command_agent, 'Runner', FakeRunner)
    monkeypatch.setattr(command_agent, 'InMemorySessionService', FakeSessionService)

    result = asyncio.run(command_agent.parse_filter_plan('show Country where Country equals India', ['Country', 'City']))

    assert 'Country' not in captured['prompt']
    assert 'FIELD_01' in captured['prompt']
    assert result['filters'][0]['column'] == 'Country'
    assert result['filters'][0]['value'] == 'India'
