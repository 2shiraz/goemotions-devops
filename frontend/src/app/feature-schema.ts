// Raw input schema for the GoEmotions sentiment model.
// The model takes ONE raw text field; cleaning + TF-IDF happen in Flask
// using the fitted objects saved from notebooks/Devops_Lab1.ipynb.
export type Feature = {
  name: string;
  label: string;
  kind: 'text';
  required: boolean;
  maxLength: number;
  placeholder?: string;
};

export const FEATURES: Feature[] = [
  {
    name: 'text',
    label: 'Comment text',
    kind: 'text',
    required: true,
    maxLength: 5000,
    placeholder: 'e.g. I am extremely proud and happy about this result!',
  },
];

export const CLASSES = ['Positive', 'Negative', 'Neutral', 'Ambiguous'] as const;

export const EXAMPLES: string[] = [
  'I am extremely proud and happy about this result!',
  'This is disappointing and makes me angry.',
  'Wait, what does that even mean?',
  'The meeting is at 3pm on Thursday.',
];
