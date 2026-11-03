"""Browser QA on authored inputs, CPU context-only; optional GPU qualification is separate."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'demo/runtime/qa';OUT.mkdir(exist_ok=True)
checks=[]
def check(page,label):
    assert page.locator('[data-testid="stException"]').count()==0,label
    assert page.evaluate('document.body.scrollWidth<=window.innerWidth+2'),label+' overflow'
    checks.append(label)
def shot(page,name):
    page.wait_for_timeout(500);page.screenshot(path=str(OUT/(name+'.png')))
def method(page,name):
    page.get_by_role('combobox',name='Method',exact=True).click();page.get_by_role('option',name=name,exact=True).click()
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-gpu'])
    page=browser.new_page(viewport={'width':1440,'height':1100})
    page.goto('http://127.0.0.1:8502',wait_until='networkidle');page.get_by_role('heading',name='Curated Research Examples').wait_for(timeout=60000)
    assert page.get_by_text('Complete curated collection',exact=False).count()==0
    assert page.get_by_text('not original benchmark scores',exact=False).count()==0
    check(page,'clean_curated');shot(page,'curated_clean')
    page.get_by_text('LiveDemo',exact=True).click();page.get_by_role('heading',name='LiveDemo',exact=True).wait_for()
    page.get_by_role('button',name='Run Evidence Pipeline',exact=True).click();page.get_by_text('Enter a non-empty query.',exact=True).wait_for();check(page,'validation_error')
    page.get_by_role('button',name='Try a sample',exact=True).click();page.get_by_role('textbox',name='Passage 5',exact=True).wait_for();check(page,'five_passages');shot(page,'live_inputs')
    # Explicitly run three passages for NONE.
    page.get_by_role('button',name='Remove',exact=True).last.click();page.get_by_role('button',name='Remove',exact=True).last.click()
    method(page,'No Evidence');page.get_by_role('button',name='Run Evidence Pipeline',exact=True).click()
    page.get_by_role('heading',name='User Query',exact=True).wait_for();page.get_by_text('No external evidence was supplied.',exact=False).wait_for();check(page,'none_three_passages');shot(page,'live_none')
    page.get_by_role('button',name='Try a sample',exact=True).click()
    for label in ['Whole passages · RAW_SKIP','Query-selected sentences','Adaptive fidelity · QBGF']:
        method(page,label);page.get_by_role('button',name='Run Evidence Pipeline',exact=True).click();page.get_by_role('heading',name='What happened in this run?',exact=True).wait_for();page.wait_for_timeout(500)
        check(page,label);shot(page,'live_'+('raw_skip' if 'RAW_SKIP' in label else 'sentences' if 'sentences' in label else 'qbgf'))
    page.get_by_role('radio',name='1024',exact=True).click();method(page,'Query-selected sentences');page.get_by_role('button',name='Run Evidence Pipeline',exact=True).click();page.wait_for_timeout(700);check(page,'sentence_1024')
    page.get_by_role('button',name='Run comparison',exact=True).click();page.wait_for_timeout(700);check(page,'method_comparison')
    page.get_by_role('heading',name='Compare with another method',exact=True).scroll_into_view_if_needed();shot(page,'live_comparison')
    page.set_viewport_size({'width':520,'height':950});check(page,'narrow_live');shot(page,'live_narrow')
    page.set_viewport_size({'width':1440,'height':1100});page.get_by_role('heading',name='User Query',exact=True).scroll_into_view_if_needed();shot(page,'live_result')
    browser.close()
(OUT/'browser_live_checks.json').write_text(json.dumps(dict(passed=checks,model_inference=False),indent=2))
print('Live browser checks passed:',len(checks))
