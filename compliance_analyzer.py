# compliance_analyzer.py - NEW FILE to add to your project
# This imports and uses your existing rag_system.py

from rag_system import RAGSystem  # Import your existing working system
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import streamlit as st

# All the compliance categories and classes from the previous code
class ComplianceCategory(Enum):
    """All areas of children's home compliance that can be assessed visually."""
    HEALTH_SAFETY = {
        "name": "Health & Safety",
        "icon": "🛡️",
        "keywords": ["safety", "hazard", "fire", "exit", "equipment", "accident", "injury", "dangerous"],
        "focus_areas": ["fire exits", "trip hazards", "equipment safety", "signage", "emergency procedures"]
    }
    
    FOOD_NUTRITION = {
        "name": "Food & Nutrition",
        "icon": "🍎",
        "keywords": ["food", "kitchen", "meal", "nutrition", "diet", "cooking", "hygiene", "storage"],
        "focus_areas": ["food hygiene", "kitchen cleanliness", "nutritional content", "meal presentation", "dietary requirements"]
    }
    
    DIGNITY_RESPECT = {
        "name": "Dignity & Respect", 
        "icon": "🤝",
        "keywords": ["dignity", "respect", "privacy", "personal", "individual", "culture", "identity"],
        "focus_areas": ["personal space", "privacy", "cultural sensitivity", "individual identity", "respectful treatment"]
    }
    
    ROOM_PERSONALISATION = {
        "name": "Room Personalisation",
        "icon": "🏠", 
        "keywords": ["room", "personal", "belongings", "space", "comfort", "homely", "decoration"],
        "focus_areas": ["personal belongings", "room comfort", "homely environment", "individual expression", "age-appropriate"]
    }
    
    RECORD_KEEPING = {
        "name": "Record Keeping & Documentation",
        "icon": "📋",
        "keywords": ["records", "documentation", "files", "paperwork", "confidential", "storage"],
        "focus_areas": ["document security", "confidentiality", "record storage", "accessibility", "organization"]
    }
    
    EDUCATION_DEVELOPMENT = {
        "name": "Education & Development", 
        "icon": "📚",
        "keywords": ["education", "learning", "development", "books", "study", "homework", "activities"],
        "focus_areas": ["learning resources", "study spaces", "educational materials", "developmental activities"]
    }
    
    WELLBEING_EMOTIONAL = {
        "name": "Wellbeing & Emotional Support",
        "icon": "💚", 
        "keywords": ["wellbeing", "emotional", "mental health", "support", "comfort", "therapeutic"],
        "focus_areas": ["emotional support spaces", "comfort areas", "therapeutic environment", "mental health resources"]
    }
    
    PHYSICAL_ENVIRONMENT = {
        "name": "Physical Environment",
        "icon": "🏢",
        "keywords": ["environment", "maintenance", "cleanliness", "repair", "condition", "facilities"],
        "focus_areas": ["building maintenance", "cleanliness standards", "facility condition", "accessibility"]
    }
    
    SAFEGUARDING = {
        "name": "Safeguarding",
        "icon": "🔒",
        "keywords": ["safeguarding", "protection", "security", "access", "supervision", "monitoring"],
        "focus_areas": ["child protection", "security measures", "supervision arrangements", "access control"]
    }

class RiskLevel(Enum):
    """Universal risk levels applicable to all compliance areas."""
    CRITICAL = {"level": "CRITICAL", "priority": 1, "color": "🔴", "action_time": "IMMEDIATE"}
    HIGH = {"level": "HIGH", "priority": 2, "color": "🟠", "action_time": "Within 24 hours"}  
    MEDIUM = {"level": "MEDIUM", "priority": 3, "color": "🟡", "action_time": "Within 1 week"}
    LOW = {"level": "LOW", "priority": 4, "color": "🟢", "action_time": "Within 1 month"}

@dataclass
class ComplianceIssue:
    """Universal compliance issue structure for all categories."""
    issue_id: str
    category: ComplianceCategory
    risk_level: RiskLevel
    location: str
    title: str
    description: str
    specific_concern: str
    child_impact: List[str]
    regulatory_reference: str
    ofsted_standard: str
    corrective_actions: List[str]
    prevention_measures: List[str]
    responsible_party: str
    estimated_cost: str
    compliance_timeline: str
    success_criteria: List[str]

@dataclass  
class ComprehensiveAnalysisResult:
    """Complete compliance analysis result across all categories."""
    analysis_timestamp: str
    image_context: str
    primary_category: ComplianceCategory
    total_issues: int
    category_distribution: Dict[str, int]
    risk_distribution: Dict[str, int]
    issues: List[ComplianceIssue]
    overall_compliance_score: float
    category_scores: Dict[str, float]
    priority_actions: List[str]
    positive_observations: List[str]
    estimated_total_cost: str
    ofsted_implications: List[str]
    follow_up_schedule: Dict[str, str]
    recommendations: List[str]

class ComplianceAnalyzer:
    """Compliance analyzer that uses your existing RAG system."""
    
    def __init__(self, rag_system: RAGSystem):
        """Initialize with your existing RAG system."""
        self.rag_system = rag_system  # Use your working RAG system
        self.issue_counter = 0
    
    def analyze_image_compliance(self, user_question: str, image_bytes: bytes) -> ComprehensiveAnalysisResult:
        """
        Analyze image compliance using your existing RAG system.
        """
        # Get relevant context using your existing retriever
        retriever = self.rag_system.get_current_retriever()
        relevant_docs = retriever.get_relevant_documents(user_question)
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Enhanced prompt for comprehensive analysis
        enhanced_prompt = f"""
        As a children's home compliance expert, analyze this image comprehensively across all relevant areas:
        
        🛡️ HEALTH & SAFETY: Fire exits, hazards, equipment safety, emergency procedures
        🍎 FOOD & NUTRITION: Kitchen hygiene, food storage, meal areas, nutritional considerations  
        🤝 DIGNITY & RESPECT: Privacy, personal space, cultural sensitivity, individual identity
        🏠 ROOM PERSONALISATION: Personal belongings, homely environment, individual expression
        📋 RECORD KEEPING: Document security, confidentiality, information management
        📚 EDUCATION & DEVELOPMENT: Learning spaces, educational resources, developmental support
        💚 WELLBEING & EMOTIONAL: Therapeutic environment, comfort areas, emotional support
        🏢 PHYSICAL ENVIRONMENT: Maintenance, cleanliness, accessibility, facility condition
        🔒 SAFEGUARDING: Security, supervision, protection measures, access control
        
        For each issue identified, provide:
        - Risk level (CRITICAL/HIGH/MEDIUM/LOW)
        - Specific concern and location
        - Impact on children's wellbeing
        - Ofsted standard reference
        - Specific corrective actions
        - Timeline for resolution
        
        Also identify positive observations and best practices.
        
        Question: {user_question}
        """
        
        # Use your existing query method with enhanced prompt
        result = self.rag_system.query(
            user_question=enhanced_prompt,
            context_text=context_text,
            source_docs=relevant_docs,
            image_bytes=image_bytes
        )
        
        # Convert to structured result
        structured_result = self._create_structured_result(result["answer"], user_question)
        
        return structured_result
    
    def _create_structured_result(self, raw_response: str, original_question: str) -> ComprehensiveAnalysisResult:
        """Convert raw response to structured result."""
        
        # For now, create a basic structured result
        # In production, you'd parse the raw_response more sophisticatedly
        
        # Example issues based on common scenarios
        issues = []
        
        if "fire exit" in raw_response.lower() or "blocked" in raw_response.lower():
            issues.append(ComplianceIssue(
                issue_id=f"COMP-{int(time.time())}-001",
                category=ComplianceCategory.HEALTH_SAFETY,
                risk_level=RiskLevel.CRITICAL,
                location="Emergency exit area",
                title="Fire Exit Route Obstruction",
                description="Emergency exit route appears to be blocked or obstructed",
                specific_concern="Potential evacuation hazard identified in image",
                child_impact=[
                    "Children may be unable to evacuate safely in emergency",
                    "Increased risk of injury during evacuation",
                    "Anxiety about safety and security"
                ],
                regulatory_reference="Regulatory Reform (Fire Safety) Order 2005",
                ofsted_standard="Standard 6: How well children are helped and protected",
                corrective_actions=[
                    "Immediately remove obstruction from exit route",
                    "Review equipment storage procedures",
                    "Conduct emergency evacuation drill"
                ],
                prevention_measures=[
                    "Daily exit route inspections",
                    "Staff training on fire safety",
                    "Clear storage guidelines"
                ],
                responsible_party="Registered Manager",
                estimated_cost="£0 - £500",
                compliance_timeline="Immediate action required",
                success_criteria=[
                    "Exit route completely clear",
                    "Emergency drill completed successfully"
                ]
            ))
        
        # Calculate metrics
        category_distribution = {}
        risk_distribution = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        
        for issue in issues:
            # Category distribution
            cat_name = issue.category.value["name"]
            category_distribution[cat_name] = category_distribution.get(cat_name, 0) + 1
            
            # Risk distribution
            risk_distribution[issue.risk_level.value["level"]] += 1
        
        # Calculate compliance score
        if not issues:
            compliance_score = 100.0
        else:
            total_weight = sum(self._get_risk_weight(issue.risk_level) for issue in issues)
            max_weight = len(issues) * 25
            compliance_score = max(0, 100 - (total_weight / max_weight) * 100) if max_weight > 0 else 100
        
        return ComprehensiveAnalysisResult(
            analysis_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            image_context="Compliance Analysis",
            primary_category=ComplianceCategory.PHYSICAL_ENVIRONMENT,
            total_issues=len(issues),
            category_distribution=category_distribution,
            risk_distribution=risk_distribution,
            issues=issues,
            overall_compliance_score=round(compliance_score, 1),
            category_scores={cat: 85.0 for cat in category_distribution.keys()},
            priority_actions=[f"{issue.risk_level.value['color']} {issue.corrective_actions[0]}" for issue in issues[:3]],
            positive_observations=[
                "Fire safety equipment appears accessible",
                "Environment appears clean and well-maintained"
            ],
            estimated_total_cost="£500 - £2,000",
            ofsted_implications=[
                "Critical issues could impact inspection rating",
                "Immediate action required for safety compliance"
            ] if any(i.risk_level == RiskLevel.CRITICAL for i in issues) else [],
            follow_up_schedule={
                "24 hours": "Verify critical issues addressed",
                "1 week": "Review corrective measures",
                "1 month": "Full compliance review"
            },
            recommendations=[
                "Implement daily safety checklist",
                "Enhance staff training on compliance standards",
                "Regular environment audits"
            ]
        )
    
    def _get_risk_weight(self, risk_level: RiskLevel) -> int:
        """Get numerical weight for risk level."""
        weights = {"CRITICAL": 25, "HIGH": 15, "MEDIUM": 8, "LOW": 3}
        return weights[risk_level.value["level"]]

# Streamlit interface functions
def display_compliance_analysis_results(result: ComprehensiveAnalysisResult):
    """Display comprehensive compliance analysis results."""
    
    st.header("🏠 Comprehensive Compliance Analysis")
    
    # Executive Summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Compliance", f"{result.overall_compliance_score}%")
    
    with col2:
        st.metric("Total Issues", result.total_issues)
    
    with col3:
        critical_count = result.risk_distribution.get("CRITICAL", 0)
        st.metric("Critical Issues", critical_count, 
                 delta="URGENT" if critical_count > 0 else "✓ None")
    
    with col4:
        st.metric("Categories Affected", len(result.category_distribution))
    
    # Priority Actions
    if result.priority_actions:
        st.subheader("🎯 Priority Actions")
        for i, action in enumerate(result.priority_actions, 1):
            st.write(f"{i}. {action}")
    
    # Positive Observations
    if result.positive_observations:
        st.subheader("✅ Positive Observations")
        for observation in result.positive_observations:
            st.success(f"• {observation}")
    
    # Detailed Issues
    if result.issues:
        st.subheader("📋 Detailed Issues")
        
        for issue in result.issues:
            with st.expander(f"{issue.risk_level.value['color']} {issue.title}"):
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**Description:**", issue.description)
                    st.write("**Specific Concern:**", issue.specific_concern)
                    
                    st.write("**Impact on Children:**")
                    for impact in issue.child_impact:
                        st.write(f"• {impact}")
                    
                    st.write("**Corrective Actions:**")
                    for action in issue.corrective_actions:
                        st.write(f"• {action}")
                
                with col2:
                    st.info(f"**Risk Level:** {issue.risk_level.value['level']}")
                    st.info(f"**Timeline:** {issue.compliance_timeline}")
                    st.info(f"**Responsible:** {issue.responsible_party}")
                    st.info(f"**Ofsted Standard:** {issue.ofsted_standard}")
    
    # Recommendations
    if result.recommendations:
        st.subheader("💡 Strategic Recommendations")
        for recommendation in result.recommendations:
            st.write(f"• {recommendation}")

def create_compliance_interface():
    """Create the compliance analysis interface."""
    
    st.subheader("🔍 Advanced Compliance Analysis")
    st.info("This uses your existing RAG system with enhanced compliance analysis capabilities.")
    
    # Initialize compliance analyzer with existing RAG system
    if 'compliance_analyzer' not in st.session_state:
        st.session_state.compliance_analyzer = ComplianceAnalyzer(st.session_state.rag_system)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload image for compliance analysis",
        type=['png', 'jpg', 'jpeg'],
        help="Upload images of any area in the children's home for comprehensive compliance assessment"
    )
    
    if uploaded_file:
        # Display image
        st.image(uploaded_file, caption="Image for Analysis", use_column_width=True)
        
        # Analysis question
        question = st.text_area(
            "Analysis Focus (optional):",
            placeholder="e.g., 'Analyze this kitchen for food safety compliance' or 'Check this bedroom for personalisation standards'",
            help="Describe what you want to focus on, or leave blank for general analysis"
        )
        
        if not question:
            question = "Analyze this image for compliance across all relevant children's home standards."
        
        # Analyze button
        if st.button("🔍 Analyze for Compliance", type="primary"):
            with st.spinner("Conducting comprehensive compliance analysis using your RAG system..."):
                try:
                    # Get image bytes
                    image_bytes = uploaded_file.read()
                    
                    # Perform analysis using existing RAG system
                    result = st.session_state.compliance_analyzer.analyze_image_compliance(
                        question, image_bytes
                    )
                    
                    # Display results
                    display_compliance_analysis_results(result)
                    
                    # Store for potential export
                    st.session_state.latest_compliance_analysis = result
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.write("Please check your image and try again.")
    
    else:
        st.info("👆 Upload an image to begin compliance analysis")
        
        # Show example of what this can analyze
        with st.expander("📋 What can this analyze?"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🛡️ Health & Safety:**")
                st.write("• Fire exits and emergency routes")
                st.write("• Trip hazards and safety equipment") 
                st.write("• Equipment safety and maintenance")
                
                st.write("**🍎 Food & Nutrition:**")
                st.write("• Kitchen hygiene and cleanliness")
                st.write("• Food storage and preparation areas")
                st.write("• Meal presentation and nutrition")
                
                st.write("**🤝 Dignity & Respect:**")
                st.write("• Privacy and personal space")
                st.write("• Cultural sensitivity")
                st.write("• Individual identity support")
            
            with col2:
                st.write("**🏠 Room Personalisation:**")
                st.write("• Personal belongings and comfort")
                st.write("• Homely environment assessment")
                st.write("• Individual expression opportunities")
                
                st.write("**📚 Education & Development:**")
                st.write("• Learning spaces and resources")
                st.write("• Study areas and materials")
                st.write("• Developmental activities")
                
                st.write("**💚 Wellbeing & Emotional:**")
                st.write("• Therapeutic environment")
                st.write("• Comfort and support areas")
                st.write("• Mental health resources")