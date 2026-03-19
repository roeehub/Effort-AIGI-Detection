"""
Curriculum learning mixin for Trainer.

Provides lesson gate functionality for curriculum learning, allowing training
to progress through different phases based on metric thresholds.
"""
from typing import Any, Dict, List, Optional


class CurriculumMixin:
    """
    Mixin that provides curriculum learning functionality for the Trainer.
    
    The lesson gate evaluates multiple conditions and determines when to
    progress to the next lesson (curriculum stage).
    
    Assumes the base class has:
    - self.config: Training configuration dict
    - self.logger: Logger instance
    """
    
    def init_curriculum(self) -> None:
        """Initialize curriculum learning state. Call from __init__."""
        self.gate_config = self.config.get('lesson_gate', {})
        self.gate_enabled = self.gate_config.get('enabled', False)
        
        if self.gate_enabled:
            self.gate_checks = self.gate_config.get('checks', [])
            self.gate_plateau_config = self.gate_config.get('plateau_check', {})
            self.gate_guardrail_config = self.gate_config.get('guardrail_check', {})

            # State tracking
            self.gate_primary_metric_history: List[float] = []
            self.gate_guardrail_start_value: Optional[float] = None

            self.logger.info("✅ Lesson Gate enabled with the following configuration:")
            self.logger.info(f"   - Checks: {len(self.gate_checks)} conditions")
            self.logger.info(f"   - Plateau Check: {self.gate_plateau_config}")
            self.logger.info(f"   - Guardrail Check: {self.gate_guardrail_config}")
        else:
            self.gate_checks = []
            self.gate_plateau_config = {}
            self.gate_guardrail_config = {}
            self.gate_primary_metric_history = []
            self.gate_guardrail_start_value = None
    
    def evaluate_lesson_gate(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Evaluate all lesson gate conditions.
        
        Args:
            metrics: Dictionary of current validation metrics
            
        Returns:
            Dictionary with:
            - 'passed': bool - whether all conditions passed
            - 'checks': list of individual check results
            - 'should_end_lesson': bool - whether to end current lesson
            - 'reason': str - reason for the decision
        """
        if not self.gate_enabled:
            return {
                'passed': False,
                'checks': [],
                'should_end_lesson': False,
                'reason': 'Lesson gate disabled'
            }
        
        check_results = []
        all_passed = True
        
        # Evaluate threshold checks
        for check in self.gate_checks:
            metric_name = check.get('metric')
            threshold = check.get('threshold')
            comparison = check.get('comparison', '>=')
            
            if metric_name not in metrics:
                self.logger.warning(f"Metric '{metric_name}' not found in validation results")
                check_results.append({
                    'metric': metric_name,
                    'passed': False,
                    'reason': 'Metric not found'
                })
                all_passed = False
                continue
            
            current_value = metrics[metric_name]
            passed = self._evaluate_comparison(current_value, threshold, comparison)
            
            check_results.append({
                'metric': metric_name,
                'value': current_value,
                'threshold': threshold,
                'comparison': comparison,
                'passed': passed
            })
            
            if not passed:
                all_passed = False
        
        # Evaluate plateau check
        plateau_result = self._check_plateau(metrics)
        if plateau_result['triggered']:
            return {
                'passed': False,
                'checks': check_results,
                'should_end_lesson': True,
                'reason': f"Plateau detected: {plateau_result['reason']}"
            }
        
        # Evaluate guardrail check
        guardrail_result = self._check_guardrail(metrics)
        if guardrail_result['violated']:
            return {
                'passed': False,
                'checks': check_results,
                'should_end_lesson': True,
                'reason': f"Guardrail violated: {guardrail_result['reason']}"
            }
        
        if all_passed:
            return {
                'passed': True,
                'checks': check_results,
                'should_end_lesson': True,
                'reason': 'All threshold conditions met'
            }
        
        return {
            'passed': False,
            'checks': check_results,
            'should_end_lesson': False,
            'reason': 'Not all conditions met yet'
        }
    
    def _evaluate_comparison(
        self,
        value: float,
        threshold: float,
        comparison: str
    ) -> bool:
        """Evaluate a comparison between a value and threshold."""
        if comparison == '>=':
            return value >= threshold
        elif comparison == '>':
            return value > threshold
        elif comparison == '<=':
            return value <= threshold
        elif comparison == '<':
            return value < threshold
        elif comparison == '==':
            return abs(value - threshold) < 1e-6
        else:
            self.logger.warning(f"Unknown comparison operator: {comparison}")
            return False
    
    def _check_plateau(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Check if training has plateaued."""
        if not self.gate_plateau_config.get('enabled', False):
            return {'triggered': False}
        
        metric_name = self.gate_plateau_config.get('metric', 'val/auc')
        patience = self.gate_plateau_config.get('patience', 5)
        min_delta = self.gate_plateau_config.get('min_delta', 0.001)
        
        if metric_name not in metrics:
            return {'triggered': False}
        
        current_value = metrics[metric_name]
        self.gate_primary_metric_history.append(current_value)
        
        if len(self.gate_primary_metric_history) < patience:
            return {'triggered': False}
        
        # Check if metric has improved in the last 'patience' evaluations
        recent_history = self.gate_primary_metric_history[-patience:]
        max_recent = max(recent_history)
        min_recent = min(recent_history)
        
        if max_recent - min_recent < min_delta:
            return {
                'triggered': True,
                'reason': f"Metric '{metric_name}' plateaued (range {max_recent - min_recent:.4f} < {min_delta})"
            }
        
        return {'triggered': False}
    
    def _check_guardrail(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Check if a guardrail metric has degraded too much."""
        if not self.gate_guardrail_config.get('enabled', False):
            return {'violated': False}
        
        metric_name = self.gate_guardrail_config.get('metric', 'val/acc')
        max_degradation = self.gate_guardrail_config.get('max_degradation', 0.05)
        
        if metric_name not in metrics:
            return {'violated': False}
        
        current_value = metrics[metric_name]
        
        # Initialize start value on first call
        if self.gate_guardrail_start_value is None:
            self.gate_guardrail_start_value = current_value
            return {'violated': False}
        
        degradation = self.gate_guardrail_start_value - current_value
        
        if degradation > max_degradation:
            return {
                'violated': True,
                'reason': f"Metric '{metric_name}' degraded by {degradation:.4f} (max allowed: {max_degradation})"
            }
        
        return {'violated': False}
    
    def reset_curriculum_state(self) -> None:
        """Reset curriculum state for a new lesson."""
        self.gate_primary_metric_history = []
        self.gate_guardrail_start_value = None
        self.logger.info("🔄 Curriculum state reset for new lesson.")
