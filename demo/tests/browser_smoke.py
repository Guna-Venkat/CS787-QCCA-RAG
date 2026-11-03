"""Run against the locally started cached app; uses existing Chrome, no downloads."""
import json,re,os
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'demo/artifacts';OUT.mkdir(exist_ok=True)
checks=[]
def clean(page,name):
    assert page.locator('[data-testid="stException"]').count()==0,name
    assert page.evaluate('document.body.scrollWidth <= window.innerWidth+2'),name+' horizontal overflow'
    checks.append(name)
def shot(page,name):
    page.wait_for_timeout(600) # Let expander/plot layout transitions settle before visual QA.
    page.screenshot(path=str(OUT/(name+'.png')))
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-gpu'])
    page=browser.new_page(viewport={'width':1440,'height':1000})
    page.goto(os.environ.get('DEMO_QA_URL','http://127.0.0.1:8501'),wait_until='networkidle')
    page.get_by_role('heading',name='Curated Research Examples').wait_for(timeout=60000)
    page.get_by_text('Researcher Mode',exact=True).click()
    page.get_by_text('Historical exports',exact=True).evaluate('(e)=>e.scrollIntoView({block:"center"})')
    page.get_by_text('Historical exports',exact=True).click()
    page.get_by_text('EXPERIMENTAL REPLAY',exact=False).wait_for(timeout=40000)
    clean(page,'lab_desktop');shot(page,'evidence_lab')
    page.get_by_text('Same query, different retrieved evidence',exact=True).click()
    page.get_by_text('Retriever Effect Explorer',exact=True).evaluate('(e)=>e.scrollIntoView({block:"start"})')
    clean(page,'retriever_query_comparison');shot(page,'retriever_explorer')
    page.get_by_role('radio',name='MiniLM',exact=True).click()
    page.get_by_text('Retrieval-only exploration:',exact=False).wait_for()
    clean(page,'missing_minilm_retrieval_only')
    page.get_by_role('radio',name='SFR',exact=True).click()
    page.get_by_text('Why whole passages sometimes cannot fit',exact=True).click()
    page.get_by_text('Why whole passages sometimes cannot fit',exact=True).evaluate('(e)=>e.scrollIntoView({block:"start"})')
    page.get_by_text('This is an exact standalone eligibility check,',exact=False).wait_for()
    clean(page,'real_source_packing');shot(page,'packing_pathology')
    page.get_by_text('What does evidence granularity mean?',exact=True).evaluate('(e)=>e.scrollIntoView({block:"start"})')
    shot(page,'granularity_explorer')
    graph=page.locator('.js-plotly-plot').last
    graph.scroll_into_view_if_needed()
    expected=graph.evaluate('(e)=>e.data[0].customdata[1][0]')
    graph.locator('.scatterlayer .trace').first.locator('.points path').nth(1).click(force=True)
    page.get_by_text('Inspect a unit · click the chart or select its ID',exact=True).wait_for()
    page.wait_for_timeout(600)
    focus=page.get_by_role('combobox',name='Inspect a unit · click the chart or select its ID',exact=True)
    assert expected == focus.input_value(), 'scatter did not focus expected saved unit'
    clean(page,'scatter_selection');shot(page,'evidence_utility')
    page.get_by_text('Research Findings',exact=True).click()
    page.get_by_text('What survived the experiments?',exact=True).wait_for()
    clean(page,'findings_desktop');shot(page,'research_findings')
    for tab in ['02 · Granularity × budget','03 · Query conditioning','04 · Fusion ablation','05 · Relevance ≠ utility','06 · Retriever effects','07 · Reader effects','08 · Decision & complexity']:
        page.get_by_role('tab',name=tab,exact=True).click();clean(page,tab)
        if tab.startswith('06'):shot(page,'retriever_findings')
        elif tab.startswith('07'):shot(page,'reader_effects')
        elif tab.startswith('08'):shot(page,'decision_protocol')
    page.get_by_text('Research Journey',exact=True).click();page.get_by_text('From adaptive context size to evidence allocation',exact=True).wait_for();clean(page,'journey');shot(page,'research_journey')
    page.set_viewport_size({'width':820,'height':1000});clean(page,'tablet_journey');shot(page,'tablet')
    narrow=browser.new_page(viewport={'width':520,'height':900})
    narrow.goto(os.environ.get('DEMO_QA_URL','http://127.0.0.1:8501'),wait_until='networkidle');narrow.get_by_role('heading',name='Curated Research Examples').wait_for();clean(narrow,'narrow_lab');shot(narrow,'narrow')
    dark=browser.new_page(viewport={'width':1440,'height':1000},color_scheme='dark')
    dark.goto(os.environ.get('DEMO_QA_URL','http://127.0.0.1:8501'),wait_until='networkidle');dark.get_by_role('heading',name='Curated Research Examples').wait_for();clean(dark,'dark_lab');shot(dark,'dark_lab')
    browser.close()
(OUT/'browser_checks.json').write_text(json.dumps({'passed':checks,'models_loaded':False,'viewport_widths':[1440,820,520]},indent=2))
print('Browser checks passed:',len(checks))
