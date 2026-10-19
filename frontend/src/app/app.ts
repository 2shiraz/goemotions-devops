import { Component, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CLASSES, EXAMPLES, FEATURES } from './feature-schema';
import { PredictionResponse, PredictionService } from './prediction.service';

@Component({
  selector: 'app-root',
  imports: [FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  readonly field = FEATURES[0];
  readonly classes = CLASSES;
  readonly examples = EXAMPLES;
  readonly deployedStatic = !['localhost', '127.0.0.1'].includes(window.location.hostname);

  readonly text = signal('');
  readonly loading = signal(false);
  readonly error = signal('');
  readonly result = signal<PredictionResponse | null>(null);

  readonly charCount = computed(() => this.text().length);
  readonly rankedScores = computed(() => {
    const scores = this.result()?.scores ?? {};
    return Object.entries(scores)
      .map(([label, score]) => ({ label, score }))
      .sort((a, b) => b.score - a.score);
  });

  private api = inject(PredictionService);

  useExample(example: string) {
    this.text.set(example);
    this.error.set('');
    this.result.set(null);
  }

  barWidth(score: number): number {
    // Map decision scores (roughly -2..+2) to a 4–100% bar for display only.
    return Math.max(4, Math.min(100, ((score + 2) / 4) * 100));
  }

  submit() {
    this.error.set('');
    this.result.set(null);
    if (this.deployedStatic) {
      this.error.set('Static GitHub Pages demo: start Angular + Flask locally for predictions.');
      return;
    }
    const value = this.text().trim();
    if (!value) {
      this.error.set(`Enter a value for ${this.field.label}`);
      return;
    }
    if (value.length > this.field.maxLength) {
      this.error.set(`${this.field.label} must be at most ${this.field.maxLength} characters`);
      return;
    }
    this.loading.set(true);
    this.api.predict({ text: value }).subscribe({
      next: (res) => {
        this.result.set(res);
        this.loading.set(false);
      },
      error: (err) => {
        this.error.set(
          `API error ${err.status || ''}: ` +
            (err.error?.detail || err.error?.error || err.message || 'Check the Flask terminal'),
        );
        this.loading.set(false);
      },
    });
  }
}
