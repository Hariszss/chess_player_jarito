import chess
import torch
import re
from typing import Optional
from chess_tournament import Player


class TransformerPlayer(Player):
    """
    GPT-2 fine-tuned on chess FEN -> UCI move data.
    Guarantees 0 fallbacks via legal-move scoring fallback.
    """

    def __init__(self, name: str = 'TransformerPlayer',
                 model_name: str = 'hariszhshsss/chess-gpt2',
                 max_tries: int = 10):
        super().__init__(name)
        from transformers import GPT2Tokenizer, GPT2LMHeadModel

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.max_tries = max_tries

    def _parse_move(self, text: str) -> Optional[str]:
        match = re.search(r'\b([a-h][1-8][a-h][1-8][qrbnQRBN]?)\b', text)
        return match.group(1).lower() if match else None

    def _score_move(self, fen: str, move: str) -> float:
        prompt = f"{fen} | {move}"
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs['input_ids'])
        return -outputs.loss.item()

    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]
        if not legal_moves:
            return None

        prompt = f"{fen} | "
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        for _ in range(self.max_tries):
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=6,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            generated = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
            move = self._parse_move(generated)
            if move and move in legal_moves:
                return move

        return max(legal_moves, key=lambda m: self._score_move(fen, m))
