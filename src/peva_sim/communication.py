"""Versioned CIB transport through simulator inboxes, not global tensors.

Payload: version:u8, dimension:u8, scale:f32, relevance:f32, code:32*i8.
42 payload bytes (336 bits), plus the simulator's separate radio header.
Relevance is a transmitted nonnegative sender score, not privileged truth.
"""
import struct
import numpy as np

PACKET = struct.Struct("<BBff32b")
MESSAGE_DIM = 32
VERSION = 1


def encode_message(message, relevance=1.0):
    x = np.asarray(message, dtype=np.float64)
    if x.shape != (MESSAGE_DIM,) or not np.isfinite(x).all():
        raise ValueError("message must be 32 finite values")
    if not np.isfinite(relevance) or not 0 <= relevance <= np.finfo(np.float32).max:
        raise ValueError("relevance must be finite, nonnegative float32")
    maximum = float(np.abs(x).max())
    scale = np.float32(max(maximum / 127, np.finfo(np.float32).tiny))
    if not np.isfinite(scale):
        raise ValueError("message exceeds representable scale")
    code = np.clip(np.rint(x / float(scale)), -127, 127).astype(np.int8)
    return PACKET.pack(VERSION, MESSAGE_DIM, float(scale), float(relevance), *code)


def decode_message(payload):
    if not isinstance(payload, bytes) or len(payload) != PACKET.size:
        raise ValueError("invalid CIB packet length/type")
    version, dimension, scale, relevance, *code = PACKET.unpack(payload)
    if version != VERSION or dimension != MESSAGE_DIM:
        raise ValueError("unsupported CIB packet schema")
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(relevance) or relevance < 0:
        raise ValueError("invalid CIB packet scale/relevance")
    message = np.asarray(code, dtype=np.float64) * scale
    if not np.isfinite(message).all() or np.abs(message).max() > np.finfo(np.float32).max:
        raise ValueError("decoded message out of range")
    return message.astype(np.float32), float(relevance)


def aggregate_inbox(inbox):
    """Return weighted mean and received count; empty/zero weight yields zero.

    Only messages actually delivered by env.step are allowed here. Sender IDs
    order the sum for deterministic permutation invariance. Duplicate senders
    are rejected rather than implicitly increasing their relevance.
    """
    entries = sorted(inbox, key=lambda pair: pair[0])
    senders = [sender for sender, _ in entries]
    if len(set(senders)) != len(senders):
        raise ValueError("duplicate sender in one communication step")
    numerator = np.zeros(MESSAGE_DIM, dtype=np.float64)
    denominator = 0.
    for _, payload in entries:
        message, relevance = decode_message(payload)
        numerator += message.astype(np.float64) * relevance
        denominator += relevance
    context = numerator / denominator if denominator > 0 else numerator
    return context.astype(np.float32), len(entries)


def received_contexts(observation):
    """Per-receiver contexts; never reads other agents' undelivered messages."""
    pairs = [aggregate_inbox(inbox) for inbox in observation["inbox"]]
    return np.stack([p[0] for p in pairs]), np.asarray([p[1] for p in pairs], dtype=np.int64)
