# Input Prompt Display Feature

## Summary
Added input frame display to show the actual prompt sent to LLMs, with highlighted instructional prompts for token-limited settings.

## Changes Made

### 1. Modified LLM Client to Return Actual Prompt

**File: `demo/llm_client.py`**

#### OpenRouterClient.call_llm_by_name (lines 120-165)
- Changed return type from `Tuple[str, int]` to `Tuple[str, int, str]`
- Now returns: `(response, token_count, actual_prompt_sent)`
- Captures the `modified_query` that was actually sent to the API
- For limited settings: includes instructional prompt
- For unlimited: returns original query unchanged

#### MockLLMClient.call_llm_by_name (lines 233-273)
- Same changes as OpenRouterClient for consistency
- Ensures mock mode behaves identically to real API mode

### 2. Updated App to Capture and Display Prompt

**File: `demo/app.py`**

#### process_query function (lines 140-170)
- Updated LLM client call to capture 3-tuple return value:
  ```python
  response, actual_tokens, actual_prompt = llm_client.call_llm_by_name(...)
  ```
- Added `actual_prompt` and `original_query` to result dictionary
- These are passed to the HTML generation for display

#### format_prompt_with_highlight function (lines 188-220)
- **New helper function** to format prompt with highlighted instructional portion
- Takes: `original_query`, `actual_prompt`, `token_limit`
- Returns: HTML string with highlighted instructional prompt
- **Logic:**
  - For limited settings (token_limit ≠ "unlimited"): Splits prompt into original query + instructional part
  - Wraps instructional part in `<span class="instructional-prompt">` for highlighting
  - For unlimited/CARROT (no instructional prompt): Returns plain escaped prompt
- Escapes all HTML characters for safe display

### 3. Added CSS Styling for Input Frame

**File: `demo/app.py`** (lines 303-323)

```css
.input-frame {
    background: #f0f8ff;           /* Light blue background */
    padding: 15px;
    border-left: 4px solid #667eea; /* Purple left border */
    margin: 15px 0;
    border-radius: 5px;
    font-family: monospace;        /* Monospace for code-like display */
    white-space: pre-wrap;         /* Preserve formatting */
}

.input-label {
    font-weight: bold;
    color: #667eea;                /* Purple color */
    margin-bottom: 10px;
}

.instructional-prompt {
    background: #fff3cd;           /* Yellow highlight background */
    padding: 5px;
    border-radius: 3px;
    color: #856404;                /* Dark yellow text */
    font-weight: bold;
}
```

### 4. Updated HTML Generation to Show Input Frame

**File: `demo/app.py`** (lines 446-449)

Added input frame display before the response section:
```html
<div style="margin-top: 20px;">
    <div class="input-label">📝 Input Prompt Sent to LLM:</div>
    <div class="input-frame">{format_prompt_with_highlight(...)}</div>
</div>
```

## Visual Result

### For R2-Router with Limited Token Setting:
- Shows original query in normal text
- Shows instructional prompt in **highlighted yellow box**:
  ```
  You have a strict budget of 150 words.
  You must answer in at most 150 words!
  Answer:
  ```

### For R2-Router with Unlimited Setting:
- Shows only the original query (no highlighting)

### For CARROT:
- Always shows only the original query (CARROT always uses unlimited)
- No instructional prompt since CARROT doesn't support limited settings

## Example Display

```
📝 Input Prompt Sent to LLM:
┌────────────────────────────────────────────────
│ What is the capital of France?
│
│ [HIGHLIGHTED] You have a strict budget of 50 words.
│ [HIGHLIGHTED] You must answer in at most 50 words!
│ [HIGHLIGHTED] Answer:
└────────────────────────────────────────────────
```

## Benefits

1. **Transparency**: Users can see exactly what prompt was sent to the LLM
2. **Debugging**: Helps verify that instructional prompts are correctly added
3. **Comparison**: Easy to compare R2-Router (with instructional prompt) vs CARROT (without)
4. **Education**: Users understand how token limits are enforced through prompts
5. **Architectural Clarity**: Visual distinction between R2-Router's curve approach and CARROT's single-point approach

## Technical Notes

- HTML escaping prevents XSS vulnerabilities
- Monospace font makes prompt structure clear
- Color scheme consistent with rest of the UI
- Works for both mock and real API modes
- Handles edge cases (empty prompts, special characters, etc.)
