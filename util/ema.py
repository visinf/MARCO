"""EMA (Exponential Moving Average) utilities for teacher-student training."""

import math


def get_ema_momentum(step, total_steps, m_start=0.99, m_end=0.9995, ramp_ratio=0.9):
    """Cosine ramp from m_start to m_end over the first ramp_ratio of training."""
    ramp_steps = int(total_steps * ramp_ratio)
    if step >= ramp_steps:
        return m_end
    cos_factor = (1 + math.cos(math.pi * step / ramp_steps)) / 2
    return m_end - (m_end - m_start) * cos_factor


def init_teacher_from_student(student, teacher):
    """Copy all parameters and buffers from student into teacher (no gradients)."""
    s = student.ddp_module.module if hasattr(student, 'ddp_module') else student
    t = teacher.ddp_module.module if hasattr(teacher, 'ddp_module') else teacher

    for p in t.parameters():
        p.requires_grad = False
    t.eval()

    s_params = dict(s.named_parameters())
    for name, t_p in t.named_parameters():
        s_p = s_params.get(name)
        if s_p is not None:
            t_p.copy_(s_p.detach().to(t_p.device, dtype=t_p.dtype, non_blocking=True))

    s_bufs = dict(s.named_buffers())
    for name, t_b in t.named_buffers():
        s_b = s_bufs.get(name)
        if s_b is not None:
            t_b.copy_(s_b.detach().to(t_b.device, dtype=t_b.dtype, non_blocking=True))
