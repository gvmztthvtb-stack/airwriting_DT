# -*- coding: utf-8 -*-
# streaming.py - Word-level CTC inference buffer control

import time
import logging
import threading
from typing import Optional, Callable, List, Dict

logger = logging.getLogger(__name__)

class StreamingInference:
    def __init__(
        self,
        ai_engine,
        debounce_time: float = 0.3,
        char_timeout: float = 0.8,
        space_timeout: float = 2.0,
    ):
        self.engine = ai_engine
        self.debounce_time = debounce_time
        self.char_timeout = char_timeout
        self.space_timeout = space_timeout

        self._strokes = []
        self._frame_count = 0
        self._max_frames = 400

        self._last_emitted_char = None
        self._last_emit_time = 0.0
        self._last_writing_time = time.time()
        self._space_emitted = False
        self._last_frame_is_writing = False
        
        self._tentative_char = None
        self._inference_running = False
        self._pending_inference = False
        self._partial_inferred = False  # 부분 추론 무한 루프 방지 플래그 추가

        self.on_text_updated = None
        self.on_clear_trajectory = None
        self._current_sentence = ""
        self._inference_lock = threading.Lock()

    def _total_frames(self) -> int:
        return sum(len(s) for s in self._strokes)

    def reset(self):
        with self._inference_lock:
            self._strokes = []
            self._tentative_char = None
            self._partial_inferred = False
            self._space_emitted = False
            self._last_emitted_char = None
            self._current_sentence = ""
        if self.on_text_updated:
            self.on_text_updated("", "", 0.0)

    def process_frame(self, frame_data: dict, is_writing: bool):
        now = time.time()

        if is_writing:
            self._last_writing_time = now
            self._space_emitted = False
            self._partial_inferred = False  # 새 획을 그릴 때는 플래그 초기화
            self._frame_count += 1

            if not self._last_frame_is_writing:
                # 이전 획이 끝난 후 경과 시간 계산
                elapsed = now - getattr(self, "_last_stroke_end_time", now)
                if elapsed > self.char_timeout and len(self._strokes) > 0:
                    logger.info(f"New character started. Finalizing tentative: '{self._tentative_char}'")
                    with self._inference_lock:
                        if self._tentative_char is not None:
                            # 텐터티브 문자를 최종 확정하고 공백 추가
                            self._current_sentence += " "
                            if self.on_text_updated:
                                self.on_text_updated(self._current_sentence, " ", 1.0)
                            self._tentative_char = None
                        self._strokes = []
                        self._partial_inferred = False
                        if self.on_clear_trajectory:
                            self.on_clear_trajectory()
                
                self._strokes.append([])
            
            self._strokes[-1].append(frame_data)
            self._last_frame_is_writing = True
            
            while self._total_frames() > self._max_frames and len(self._strokes) > 1:
                self._strokes.pop(0)

        else:
            if self._last_frame_is_writing:
                self._last_stroke_end_time = now
            self._last_frame_is_writing = False
            duration_since_writing = now - self._last_writing_time
            
            # 1. 단일 글자 완성 판정 (char_timeout) -> 1회만 부분 추론을 날리도록 partial_inferred 플래그로 락업
            if duration_since_writing > self.char_timeout and len(self._strokes) > 0 and not self._partial_inferred:
                total = self._total_frames()
                if total > 15:
                    if not self._inference_running:
                        self._partial_inferred = True  # 루핑 방어 잠금
                        self._run_inference(is_partial=True)
                else:
                    self._strokes = []
                    self._tentative_char = None

            # 2. 단어 완성 및 띄어쓰기 판정 (space_timeout)
            if duration_since_writing > self.space_timeout and not self._space_emitted:
                if len(self._strokes) > 0:
                    if not self._inference_running:
                        self._run_inference(is_partial=False)
                self._space_emitted = True
                self._last_emitted_char = None
                self._partial_inferred = False

    def _run_inference(self, is_partial: bool = False):
        if not self.engine or len(self._strokes) == 0:
            return
        if self._inference_running:
            self._pending_inference = True
            return

        self._inference_running = True
        self._pending_inference = False
        
        session_strokes = [list(s) for s in self._strokes if len(s) > 2]
        if not is_partial:
            self._strokes = []
        
        if not session_strokes:
            self._inference_running = False
            return
        
        n_strokes = len(session_strokes)
        n_frames = sum(len(s) for s in session_strokes)
        logger.info(f"Inference (partial={is_partial}): {n_strokes} strokes, {n_frames} frames")

        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, self._do_predict_and_emit, session_strokes, is_partial)
        except RuntimeError:
            self._do_predict_and_emit(session_strokes, is_partial)

    def _do_predict_and_emit(self, session_strokes, is_partial: bool):
        try:
            result = self.engine.predict(session_strokes)

            if not result:
                with self._inference_lock:
                    if not is_partial and self._tentative_char is not None:
                        for _ in range(len(self._tentative_char)):
                            self._emit_char("<ERASE>")
                        self._tentative_char = None
                return

            if isinstance(result, tuple) and len(result) == 2:
                predicted_word, conf = result
            else:
                predicted_word = result
                conf = 1.0

            if not predicted_word:
                with self._inference_lock:
                    if not is_partial and self._tentative_char is not None:
                        for _ in range(len(self._tentative_char)):
                            self._emit_char("<ERASE>")
                        self._tentative_char = None
                return

            if is_partial and conf < 0.65:
                return

            char = predicted_word
            now = time.time()
            chars_to_emit = []

            with self._inference_lock:
                if is_partial:
                    if self._tentative_char != char:
                        if self._tentative_char is not None:
                            # 텐터티브 텍스트의 전체 길이만큼 지우기 수행
                            for _ in range(len(self._tentative_char)):
                                chars_to_emit.append("<ERASE>")
                        self._tentative_char = char
                        chars_to_emit.append(char)
                        self._last_emitted_char = char
                        self._last_emit_time = now
                else:
                    if self._tentative_char is not None:
                        if self._tentative_char != char:
                            # 텐터티브 텍스트의 전체 길이만큼 지우기 수행
                            for _ in range(len(self._tentative_char)):
                                chars_to_emit.append("<ERASE>")
                            chars_to_emit.append(char)
                        self._tentative_char = None
                        self._last_emitted_char = char
                        self._last_emit_time = now
                    else:
                        is_duplicate = (char == self._last_emitted_char) and \
                                       ((now - self._last_emit_time) < self.debounce_time)
                        if not is_duplicate:
                            self._last_emitted_char = char
                            self._last_emit_time = now
                            chars_to_emit.append(char)
                    
                    # 최종 확정된 단어 뒤에 띄어쓰기(공백) 추가
                    chars_to_emit.append(" ")

            for c in chars_to_emit:
                self._emit_char(c, conf)
        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
        finally:
            self._inference_running = False
            
            if self._pending_inference and len(self._strokes) > 0:
                self._pending_inference = False
                total = self._total_frames()
                if total > 15:
                    self._run_inference(is_partial=False)

    def _emit_char(self, char: str, confidence: float = 0.0):
        if char == "<ERASE>":
            if len(self._current_sentence) > 0:
                self._current_sentence = self._current_sentence[:-1]
            logger.info(f"Stream Erased -> Sentence: '{self._current_sentence}'")
        else:
            self._current_sentence += char
            logger.info(f"Stream Emitted: '{char}' ({confidence:.0%}) -> Sentence: '{self._current_sentence}'")
            
        if self.on_text_updated:
            self.on_text_updated(self._current_sentence, char, confidence)

    def reset(self):
        self._strokes = []
        self._frame_count = 0
        self._current_sentence = ""
        self._last_emitted_char = None
        self._last_emit_time = 0.0
        self._last_writing_time = time.time()
        self._space_emitted = False
        self._last_frame_is_writing = False
        self._tentative_char = None
        self._inference_running = False
        self._pending_inference = False
        self._partial_inferred = False
        if self.on_text_updated:
            self.on_text_updated("", "", 0.0)
