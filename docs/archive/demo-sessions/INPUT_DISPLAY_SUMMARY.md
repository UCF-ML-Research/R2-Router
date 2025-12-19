# Input Prompt Display - Quick Summary

## What Changed

Added visual display of the actual prompt sent to LLMs, with highlighted instructional prompts for token-limited settings.

## Files Modified

1. **`demo/llm_client.py`** (lines 120-165, 233-273)
   - Modified both `OpenRouterClient` and `MockLLMClient`
   - Changed `call_llm_by_name()` to return 3-tuple: `(response, token_count, actual_prompt)`
   - Captures the modified query that includes instructional prompt

2. **`demo/app.py`** (lines 140-170, 188-220, 303-323, 446-449)
   - Updated `process_query()` to capture actual_prompt from LLM client
   - Added `format_prompt_with_highlight()` helper function
   - Added CSS styling for input frame and highlighted instructional prompt
   - Added input frame display in HTML generation

## Visual Features

### Input Frame Styling
- Light blue background with purple left border
- Monospace font for code-like display
- Label: "📝 Input Prompt Sent to LLM:"

### Instructional Prompt Highlighting
- Yellow background (`#fff3cd`) with dark text
- Applied to the budget instruction part:
  ```
  You have a strict budget of {N} words.
  You must answer in at most {N} words!
  Answer:
  ```

## Display Behavior

| Method | Token Limit | Display |
|--------|-------------|---------|
| CoRE | Limited (e.g., 150) | Original query + **highlighted instructional prompt** |
| CoRE | Unlimited | Original query only |
| CARROT | N/A (always unlimited) | Original query only |

## Key Benefits

✅ **Transparency**: See exactly what was sent to the LLM
✅ **Debugging**: Verify instructional prompts are correct
✅ **Comparison**: Visual difference between CoRE and CARROT
✅ **Education**: Understand how token budgets work

## Example

**For CoRE with 50-word limit:**
```
Original query text here

[HIGHLIGHTED IN YELLOW]
You have a strict budget of 50 words.
You must answer in at most 50 words!
Answer:
```

**For CARROT or unlimited:**
```
Original query text here
```

---
*This enhancement addresses the user's request to show input frames with highlighted instructional prompts for both CoRE and CARROT.*
