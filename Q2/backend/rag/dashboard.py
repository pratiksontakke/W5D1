import asyncio
import json
from typing import Dict, List, Any
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from loguru import logger

from evaluation import evaluator
from rag_engine import rag_engine
from chroma_utils import chroma_manager

class MetricsDashboard:
    """Simple dashboard for visualizing RAG system metrics"""
    
    def __init__(self):
        self.output_dir = Path("./dashboard_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    async def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive system report"""
        logger.info("Generating comprehensive dashboard report...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_health": await self._get_system_health(),
            "performance_metrics": await self._get_performance_metrics(),
            "evaluation_results": await self._get_evaluation_results(),
            "database_stats": self._get_database_stats(),
            "usage_analytics": self._get_usage_analytics()
        }
        
        # Save report
        report_path = self.output_dir / f"comprehensive_report_{int(datetime.now().timestamp())}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to {report_path}")
        return report
    
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get system health status"""
        try:
            health = await rag_engine.health_check()
            return {
                "overall_healthy": health.get("overall", False),
                "components": health,
                "checked_at": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return {"error": str(e)}
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            # Run a quick evaluation to get current metrics
            test_cases = evaluator.test_generator.generate_test_cases()[:10]  # Quick test
            results = await evaluator.run_batch_evaluation(test_cases)
            
            return {
                "overall_accuracy": results.overall_accuracy,
                "intent_accuracy": results.intent_accuracy,
                "average_metrics": {
                    "faithfulness": results.average_metrics.faithfulness,
                    "answer_relevancy": results.average_metrics.answer_relevancy,
                    "context_precision": results.average_metrics.context_precision,
                    "context_recall": results.average_metrics.context_recall,
                    "response_time": results.average_metrics.response_time,
                    "token_usage": results.average_metrics.token_usage
                },
                "test_cases_count": len(test_cases)
            }
        except Exception as e:
            logger.error(f"Performance metrics error: {e}")
            return {"error": str(e)}
    
    async def _get_evaluation_results(self) -> Dict[str, Any]:
        """Get detailed evaluation results"""
        try:
            # Run full evaluation
            results = await evaluator.run_batch_evaluation()
            
            # Process results for dashboard
            intent_performance = {}
            for intent in ["technical", "billing", "features"]:
                intent_results = [r for r in results.results if r.test_case.expected_intent.value == intent]
                if intent_results:
                    intent_performance[intent] = {
                        "total_cases": len(intent_results),
                        "passed_cases": sum(1 for r in intent_results if r.passed),
                        "average_confidence": sum(r.response.confidence for r in intent_results) / len(intent_results),
                        "average_response_time": sum(r.response.response_time for r in intent_results) / len(intent_results)
                    }
            
            return {
                "total_evaluation": {
                    "total_cases": results.total_cases,
                    "passed_cases": results.passed_cases,
                    "overall_accuracy": results.overall_accuracy
                },
                "intent_performance": intent_performance,
                "evaluation_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Evaluation results error: {e}")
            return {"error": str(e)}
    
    def _get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            stats = chroma_manager.get_database_stats()
            return {
                "total_collections": stats.get("total_collections", 0),
                "total_documents": stats.get("total_documents", 0),
                "collections": stats.get("collections", {}),
                "checked_at": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Database stats error: {e}")
            return {"error": str(e)}
    
    def _get_usage_analytics(self) -> Dict[str, Any]:
        """Get usage analytics (mock data for demonstration)"""
        # In a real implementation, this would pull from actual usage logs
        return {
            "daily_queries": {
                "today": 245,
                "yesterday": 198,
                "last_week_avg": 156
            },
            "popular_intents": {
                "technical": 45,
                "billing": 30,
                "features": 25
            },
            "response_time_trends": {
                "avg_this_week": 1.2,
                "avg_last_week": 1.4,
                "improvement": "12%"
            }
        }
    
    def create_performance_charts(self, evaluation_results: Dict[str, Any]):
        """Create performance visualization charts"""
        try:
            # Intent accuracy chart
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Intent Classification Accuracy
            intent_data = evaluation_results.get("intent_performance", {})
            if intent_data:
                intents = list(intent_data.keys())
                accuracies = [intent_data[intent]["passed_cases"] / intent_data[intent]["total_cases"] 
                             for intent in intents]
                
                ax1.bar(intents, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                ax1.set_title('Intent Classification Accuracy', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Accuracy')
                ax1.set_ylim(0, 1)
                
                # Add value labels on bars
                for i, v in enumerate(accuracies):
                    ax1.text(i, v + 0.01, f'{v:.2%}', ha='center', va='bottom')
            
            # 2. Response Time Distribution
            if intent_data:
                response_times = [intent_data[intent]["average_response_time"] for intent in intents]
                ax2.bar(intents, response_times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                ax2.set_title('Average Response Time by Intent', fontsize=14, fontweight='bold')
                ax2.set_ylabel('Response Time (seconds)')
                
                # Add value labels
                for i, v in enumerate(response_times):
                    ax2.text(i, v + 0.01, f'{v:.2f}s', ha='center', va='bottom')
            
            # 3. Confidence Distribution
            if intent_data:
                confidences = [intent_data[intent]["average_confidence"] for intent in intents]
                ax3.bar(intents, confidences, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                ax3.set_title('Average Confidence by Intent', fontsize=14, fontweight='bold')
                ax3.set_ylabel('Confidence Score')
                ax3.set_ylim(0, 1)
                
                # Add value labels
                for i, v in enumerate(confidences):
                    ax3.text(i, v + 0.01, f'{v:.2%}', ha='center', va='bottom')
            
            # 4. Overall System Performance
            total_eval = evaluation_results.get("total_evaluation", {})
            if total_eval:
                labels = ['Passed', 'Failed']
                sizes = [total_eval.get("passed_cases", 0), 
                        total_eval.get("total_cases", 0) - total_eval.get("passed_cases", 0)]
                colors = ['#4ECDC4', '#FF6B6B']
                
                ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax4.set_title('Overall Test Results', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Save chart
            chart_path = self.output_dir / f"performance_charts_{int(datetime.now().timestamp())}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Performance charts saved to {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            logger.error(f"Chart creation error: {e}")
            return None
    
    def create_metrics_table(self, performance_metrics: Dict[str, Any]) -> str:
        """Create a formatted metrics table"""
        try:
            avg_metrics = performance_metrics.get("average_metrics", {})
            
            table_data = {
                "Metric": [
                    "Faithfulness",
                    "Answer Relevancy", 
                    "Context Precision",
                    "Context Recall",
                    "Response Time",
                    "Token Usage"
                ],
                "Value": [
                    f"{avg_metrics.get('faithfulness', 0):.3f}",
                    f"{avg_metrics.get('answer_relevancy', 0):.3f}",
                    f"{avg_metrics.get('context_precision', 0):.3f}",
                    f"{avg_metrics.get('context_recall', 0):.3f}",
                    f"{avg_metrics.get('response_time', 0):.3f}s",
                    f"{avg_metrics.get('token_usage', 0)}"
                ],
                "Target": [
                    "≥ 0.85",
                    "≥ 0.80",
                    "≥ 0.75",
                    "≥ 0.70",
                    "< 2.0s",
                    "< 1000"
                ]
            }
            
            df = pd.DataFrame(table_data)
            
            # Save as CSV
            table_path = self.output_dir / f"metrics_table_{int(datetime.now().timestamp())}.csv"
            df.to_csv(table_path, index=False)
            
            logger.info(f"Metrics table saved to {table_path}")
            return str(table_path)
            
        except Exception as e:
            logger.error(f"Table creation error: {e}")
            return None
    
    def generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate an HTML report"""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>RAG System Dashboard</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                    .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }}
                    .healthy {{ color: green; }}
                    .unhealthy {{ color: red; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🚀 RAG System Dashboard</h1>
                    <p>Generated on: {report_data.get('timestamp', 'N/A')}</p>
                </div>
                
                <div class="section">
                    <h2>System Health</h2>
                    <div class="metric">
                        <strong>Overall Status:</strong> 
                        <span class="{'healthy' if report_data.get('system_health', {}).get('overall_healthy', False) else 'unhealthy'}">
                            {'✅ Healthy' if report_data.get('system_health', {}).get('overall_healthy', False) else '❌ Unhealthy'}
                        </span>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Performance Metrics</h2>
                    <div class="metric">
                        <strong>Overall Accuracy:</strong> {report_data.get('performance_metrics', {}).get('overall_accuracy', 0):.2%}
                    </div>
                    <div class="metric">
                        <strong>Avg Response Time:</strong> {report_data.get('performance_metrics', {}).get('average_metrics', {}).get('response_time', 0):.2f}s
                    </div>
                    <div class="metric">
                        <strong>Faithfulness:</strong> {report_data.get('performance_metrics', {}).get('average_metrics', {}).get('faithfulness', 0):.3f}
                    </div>
                </div>
                
                <div class="section">
                    <h2>Database Statistics</h2>
                    <div class="metric">
                        <strong>Total Collections:</strong> {report_data.get('database_stats', {}).get('total_collections', 0)}
                    </div>
                    <div class="metric">
                        <strong>Total Documents:</strong> {report_data.get('database_stats', {}).get('total_documents', 0)}
                    </div>
                </div>
                
                <div class="section">
                    <h2>Intent Performance</h2>
                    <table>
                        <tr>
                            <th>Intent</th>
                            <th>Total Cases</th>
                            <th>Passed Cases</th>
                            <th>Accuracy</th>
                            <th>Avg Confidence</th>
                        </tr>
            """
            
            # Add intent performance rows
            intent_performance = report_data.get('evaluation_results', {}).get('intent_performance', {})
            for intent, data in intent_performance.items():
                accuracy = data['passed_cases'] / data['total_cases'] if data['total_cases'] > 0 else 0
                html_content += f"""
                        <tr>
                            <td>{intent.title()}</td>
                            <td>{data['total_cases']}</td>
                            <td>{data['passed_cases']}</td>
                            <td>{accuracy:.2%}</td>
                            <td>{data['average_confidence']:.2%}</td>
                        </tr>
                """
            
            html_content += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>Usage Analytics</h2>
                    <div class="metric">
                        <strong>Daily Queries:</strong> {daily_queries}
                    </div>
                    <div class="metric">
                        <strong>Response Time Improvement:</strong> {improvement}
                    </div>
                </div>
                
                <footer style="margin-top: 40px; text-align: center; color: #666;">
                    <p>RAG Customer Support System - Dashboard Report</p>
                </footer>
            </body>
            </html>
            """.format(
                daily_queries=report_data.get('usage_analytics', {}).get('daily_queries', {}).get('today', 0),
                improvement=report_data.get('usage_analytics', {}).get('response_time_trends', {}).get('improvement', 'N/A')
            )
            
            # Save HTML report
            html_path = self.output_dir / f"dashboard_report_{int(datetime.now().timestamp())}.html"
            with open(html_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"HTML report saved to {html_path}")
            return str(html_path)
            
        except Exception as e:
            logger.error(f"HTML report generation error: {e}")
            return None
    
    async def run_full_dashboard(self):
        """Run complete dashboard generation"""
        logger.info("Starting full dashboard generation...")
        
        try:
            # Generate comprehensive report
            report_data = await self.generate_comprehensive_report()
            
            # Create visualizations
            chart_path = self.create_performance_charts(report_data.get('evaluation_results', {}))
            table_path = self.create_metrics_table(report_data.get('performance_metrics', {}))
            html_path = self.generate_html_report(report_data)
            
            # Summary
            summary = {
                "dashboard_completed": True,
                "timestamp": datetime.now().isoformat(),
                "outputs": {
                    "comprehensive_report": str(self.output_dir / "comprehensive_report_*.json"),
                    "performance_charts": chart_path,
                    "metrics_table": table_path,
                    "html_report": html_path
                },
                "summary": {
                    "system_healthy": report_data.get('system_health', {}).get('overall_healthy', False),
                    "overall_accuracy": report_data.get('performance_metrics', {}).get('overall_accuracy', 0),
                    "total_documents": report_data.get('database_stats', {}).get('total_documents', 0)
                }
            }
            
            logger.info("Dashboard generation completed successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Dashboard generation error: {e}")
            return {"error": str(e)}

# Global dashboard instance
dashboard = MetricsDashboard()

# CLI interface for dashboard
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        asyncio.run(dashboard.run_full_dashboard())
    else:
        print("Usage: python dashboard.py run")
        print("This will generate a comprehensive dashboard report with visualizations.") 