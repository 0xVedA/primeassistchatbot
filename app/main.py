"""
main.py – RNN-powered Support Chatbot
======================================
Uses a locally trained LSTM model for intent classification.
No API calls required.
"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from app.rnn_matcher import RNNMatcher

console = Console()

# ─── Exit keywords (multilingual) ────────────────────────────────────────────
EXIT_WORDS = {
    "exit", "quit", "bye", "goodbye",
    "ende", "beenden",
    "salir", "quitter",
}

CONFIDENCE_THRESHOLD = 0.40   # below this → say "I'm not sure"

def main():
    console.print(Panel(
        "[bold green]🛒 E-Commerce Support Bot[/bold green]\n"
        "[dim]Powered by a local RNN model  •  type 'exit' to quit[/dim]",
        expand=False,
    ))

    matcher = RNNMatcher(model_dir="model")
    console.print("[green]✓[/green] Model loaded\n")

    while True:
        user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
        if not user_input:
            continue
        if user_input.lower() in EXIT_WORDS:
            console.print("[bold]Bot:[/bold] Bye! 👋")
            break

        intent, response, confidence = matcher.predict(user_input)

        if confidence < CONFIDENCE_THRESHOLD:
            console.print(
                f"[bold]Bot:[/bold] I'm not sure I understand. "
                f"Could you rephrase your question?\n"
                f"[dim](best guess: {intent}  conf={confidence:.2f})[/dim]\n"
            )
        else:
            console.print(
                f"[bold]Bot:[/bold] {response}\n"
                f"[dim](intent: {intent}  conf={confidence:.2f})[/dim]\n"
            )


if __name__ == "__main__":
    main()