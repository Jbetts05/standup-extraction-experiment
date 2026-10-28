"""Task board over a series of stand-ups: create cards, then update them from later meetings.

    simulate   sample a run of consecutive stand-ups and what happens to each card
    render     write each day's transcript (template offline, or an LLM writer)
    pipeline   step 1 = the existing extractor + matcher (new cards); step 2 = update check on open cards
    evaluate   update / false-close / duplicate / board-accuracy scores
    export     build docs/04-board-pipeline/mock-board/index.html
"""
