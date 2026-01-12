INSTRUCTION_SYSTEM = '''
You are a helpful assistant for pictographic stroke reconstruction.
'''

INSTRUCTION_USER = '''
Your task is to reconstruct the strokes of a black-and-white pictographic character based on the input image, which is defined on a normalized coordinate system where both x and y range from 0 to 1.

## Format

Your output must strictly follow the following format:

```
<stroke>[[x1,y1],[x2,y2]]</stroke><stroke>...</stroke>...
```

## Requirements

1. Each stroke is represented as a line consisting of 2 points. The entire reconstruction may contain no more than 50 strokes.
2. All coordinates must lie within [0, 1].
3. Strokes must approximate the visible trajectory of the character's brush or outline.
4. Avoid producing strokes that retrace or duplicate previous strokes.
5. No natural language, explanations, descriptions, or comments may appear in the output.
6. Use as few strokes as possible while capturing the complete structure. Prefer one multi-point stroke over multiple shorter strokes.
7. Avoid overfitting noise or artifacts from the binary image.
'''