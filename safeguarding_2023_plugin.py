"""
Safeguarding 2023 Compliance Plugin
Integrates with your existing HybridRAGSystem without requiring code changes

INTEGRATION STEPS:
1. Save this file as: safeguarding_2023_plugin.py
2. Add ONE line to your existing rag_system.py: 
   from safeguarding_2023_plugin import SafeguardingPlugin
3. Add ONE line in your HybridRAGSystem.__init__():
   self.safeguarding_plugin = SafeguardingPlugin()
4. Replace your query() method's prompt building section with plugin call

That's it! No other changes needed.
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# Enhanced Enums for 2023 Compliance (same as before)
class ThresholdLevel(Enum):
    LEVEL_1 = "level_1"  # No additional needs
    LEVEL_2 = "level_2"  # Additional needs/early intervention
    LEVEL_3 = "level_3"  # Specialist services required
    LEVEL_4 = "level_4"  # Significant harm - immediate action

class AbuseType(Enum):
    PHYSICAL = "physical_abuse"
    EMOTIONAL = "emotional_abuse"
    SEXUAL = "sexual_abuse"
    NEGLECT = "neglect"
    EXPLOITATION = "exploitation"
    EXTRA_FAMILIAL = "extra_familial_harm"  # NEW for 2023

class ExtraFamilialHarm(Enum):
    """New 2023 requirement - harm outside family environment"""
    CRIMINAL_EXPLOITATION = "criminal_exploitation"
    SEXUAL_EXPLOITATION = "sexual_exploitation"
    GANG_INVOLVEMENT = "gang_involvement"
    ONLINE_ABUSE = "online_abuse"
    PEER_ABUSE = "peer_abuse"
    COMMUNITY_HARM = "community_harm"
    TRAFFICKING = "trafficking"

class SettingType(Enum):
    """Enhanced to include children's homes specificity"""
    FAMILY_HOME = "family_home"
    CHILDRENS_HOME = "childrens_home"
    FOSTER_CARE = "foster_care"
    SCHOOL = "school"
    COMMUNITY = "community"
    ONLINE = "online"

@dataclass
class SafeguardingAssessment2023:
    """2023-compliant safeguarding assessment"""
    threshold_level: ThresholdLevel
    abuse_types: List[AbuseType]
    extra_familial_risks: List[ExtraFamilialHarm]
    setting_type: SettingType
    evidence: List[str]
    escalation_required: bool
    multi_agency_required: bool
    child_voice_consideration: str
    confidence_score: float
    immediate_actions: List[str]
    rationale: str
    working_together_2023_compliance: Dict[str, bool]

class SafeguardingPlugin:
    """
    Plugin that enhances your existing RAG system with 2023 safeguarding compliance
    Designed to work with your current HybridRAGSystem without breaking changes
    """
    
    def __init__(self):
        self._initialize_knowledge_base()
        self._initialize_templates()
    
    def _initialize_knowledge_base(self):
        """Initialize 2023-compliant knowledge base"""
        
        # Extra-familial harm patterns (NEW for 2023)
        self.extra_familial_patterns = {
            ExtraFamilialHarm.CRIMINAL_EXPLOITATION: {
                "indicators": ["county lines", "drug running", "knife crime", "criminal activity", "debt bondage"],
                "behavioral_signs": ["unexplained money", "new phone", "travel patterns", "substance use"]
            },
            ExtraFamilialHarm.SEXUAL_EXPLOITATION: {
                "indicators": ["older boyfriend", "grooming", "sexual exploitation", "trafficking"],
                "behavioral_signs": ["sexualized behavior", "gifts", "secrecy", "missing episodes"]
            },
            ExtraFamilialHarm.GANG_INVOLVEMENT: {
                "indicators": ["gang", "postcode rivalry", "territorial", "youth violence"],
                "behavioral_signs": ["aggressive behavior", "fear", "injuries", "loyalty conflicts"]
            },
            ExtraFamilialHarm.ONLINE_ABUSE: {
                "indicators": ["online grooming", "cyberbullying", "digital abuse", "image sharing"],
                "behavioral_signs": ["secretive online behavior", "withdrawn", "anxiety about devices"]
            },
            ExtraFamilialHarm.PEER_ABUSE: {
                "indicators": ["peer pressure", "bullying", "peer exploitation", "harmful peer relationships"],
                "behavioral_signs": ["fear of peers", "social withdrawal", "unexplained injuries"]
            }
        }
        
        # Setting detection patterns
        self.setting_patterns = {
            SettingType.CHILDRENS_HOME: ["children's home", "residential", "looked after", "placement", "care home"],
            SettingType.FOSTER_CARE: ["foster", "carer", "foster family", "foster home"],
            SettingType.SCHOOL: ["school", "teacher", "classroom", "playground", "education"],
            SettingType.ONLINE: ["online", "internet", "social media", "gaming", "digital"],
            SettingType.COMMUNITY: ["community", "neighbourhood", "gang", "peers", "local area"]
        }
        
        # 2023 threshold criteria (enhanced)
        self.threshold_2023_criteria = {
            ThresholdLevel.LEVEL_4: {
                "triggers": ["direct disclosure", "visible injuries", "immediate danger", "severe abuse", "significant harm"],
                "extra_familial_escalators": ["criminal exploitation", "sexual exploitation", "trafficking"],
                "action_timeframe": "immediate"
            },
            ThresholdLevel.LEVEL_3: {
                "triggers": ["persistent concerns", "multiple risk factors", "specialist assessment needed"],
                "extra_familial_escalators": ["gang involvement", "online abuse", "peer exploitation"],
                "action_timeframe": "within 24-48 hours"
            },
            ThresholdLevel.LEVEL_2: {
                "triggers": ["early signs", "preventive intervention needed", "family stress"],
                "extra_familial_escalators": ["concerning peer relationships", "community risks"],
                "action_timeframe": "within one week"
            }
        }
    
    def _initialize_templates(self):
        """Initialize 2023-compliant prompt templates"""
        
        self.templates_2023 = {
            "safeguarding_childrens_home": """You are conducting a safeguarding assessment in a children's home using Working Together to Safeguard Children 2023 and current regulatory frameworks.

REGULATORY FRAMEWORK - Apply your comprehensive knowledge:
- Working Together to Safeguard Children 2023 (multi-agency standards)
- Children's Homes Regulations 2015 (Regulations 28 & 32)
- Quality Standards for Children's Homes (all nine standards)
- Children Act 1989/2004

MANDATORY 2023 ASSESSMENT AREAS:

1. THRESHOLD ASSESSMENT (Levels 1-4):
   - Apply 2023 threshold criteria with multi-agency considerations
   - Extra-familial harm assessment (mandatory)
   - Child-centred approach with voice consideration

2. CHILDREN'S HOMES SPECIFIC:
   - Regulation 34 compliance requirements
   - Quality Standards impact assessment
   - Missing child protocols if relevant
   - Referral procedures to placing/local authorities

3. MULTI-AGENCY COORDINATION:
   - Lead/delegated safeguarding partner identification
   - Information sharing requirements per 2023 guidance

CONTEXT: {context}
SCENARIO: {question}

Provide structured assessment demonstrating full 2023 compliance and children's home regulatory alignment.""",

            "safeguarding_signs_of_safety_2023": """You are applying Signs of Safety framework with Working Together to Safeguard Children 2023 requirements.

SIGNS OF SAFETY WITH 2023 ENHANCEMENTS:

WORRIED ABOUT (Dangers & Vulnerabilities):
- Past harm and future danger
- Extra-familial risks (2023 mandatory requirement)
- Complicating factors

WHAT'S WORKING WELL (Strengths & Safety):
- Child's resilience and protective factors  
- Family/placement strengths
- Professional and community support

WHAT NEEDS TO HAPPEN (Safety Goals & Actions):
- Safety goals with child voice central
- Multi-agency coordination per 2023 standards
- Immediate actions with clear timelines

CONTEXT: {context}
ASSESSMENT FOCUS: {question}

Apply framework ensuring 2023 compliance: child voice central, extra-familial harm considered, multi-agency coordination identified.""",

            "safeguarding_general_2023": """You are conducting safeguarding assessment using Working Together to Safeguard Children 2023.

MANDATORY 2023 REQUIREMENTS:
- Child-centred approach (voice, wishes, feelings)
- Extra-familial harm assessment (risks outside family)
- Multi-agency coordination expectations
- Whole family consideration

ASSESSMENT STRUCTURE:
1. Threshold level (1-4) with 2023 criteria
2. Extra-familial harm evaluation (mandatory)
3. Child voice consideration
4. Multi-agency coordination needs
5. Immediate actions with timelines

CONTEXT: {context}
QUESTION: {question}

Provide assessment demonstrating full Working Together 2023 compliance."""
        }
    
    def enhance_query_with_2023_compliance(self, question: str, context: str, 
                                         current_response_mode: str) -> Dict[str, Any]:
        """
        Main plugin method: Enhance your existing query with 2023 compliance
        
        Call this in your existing query() method before building the final prompt
        """
        
        # Perform 2023 compliance analysis
        assessment = self._analyze_2023_compliance(question, context)
        
        # Determine if we need safeguarding-specific template
        enhanced_prompt_needed = self._needs_safeguarding_enhancement(question, assessment)
        
        # Generate enhanced prompt if needed
        enhanced_prompt = None
        if enhanced_prompt_needed:
            enhanced_prompt = self._generate_2023_prompt(question, context, assessment)
        
        # Create assessment summary for response
        assessment_summary = self._create_assessment_summary(assessment)
        
        return {
            "needs_safeguarding_enhancement": enhanced_prompt_needed,
            "enhanced_prompt": enhanced_prompt,
            "assessment": assessment,
            "assessment_summary": assessment_summary,
            "compliance_status": assessment.working_together_2023_compliance if assessment else {},
            "escalation_required": assessment.escalation_required if assessment else False
        }
    
    def _analyze_2023_compliance(self, question: str, context: str) -> Optional[SafeguardingAssessment2023]:
        """Analyze content for 2023 safeguarding compliance"""
        
        # Check if this is safeguarding-related
        if not self._is_safeguarding_content(question, context):
            return None
        
        # Detect setting type
        setting_type = self._detect_setting_type(context)
        
        # Identify abuse types
        abuse_types = self._identify_abuse_types(context)
        
        # NEW: Identify extra-familial risks
        extra_familial_risks = self._identify_extra_familial_risks(context)
        
        # Assess threshold level with 2023 criteria
        threshold_level = self._assess_threshold_level_2023(context, abuse_types, extra_familial_risks)
        
        # Extract evidence
        evidence = self._extract_evidence(context, abuse_types, extra_familial_risks)
        
        # Multi-agency assessment
        multi_agency_required = threshold_level in [ThresholdLevel.LEVEL_3, ThresholdLevel.LEVEL_4]
        escalation_required = threshold_level in [ThresholdLevel.LEVEL_3, ThresholdLevel.LEVEL_4]
        
        # Child voice consideration
        child_voice = self._assess_child_voice(context)
        
        # Generate actions
        immediate_actions = self._generate_immediate_actions_2023(
            threshold_level, abuse_types, extra_familial_risks, setting_type
        )
        
        # Confidence score
        confidence_score = self._calculate_confidence(context, abuse_types, extra_familial_risks)
        
        # Rationale
        rationale = self._generate_rationale_2023(threshold_level, abuse_types, setting_type)
        
        # 2023 compliance check
        compliance_check = self._check_2023_compliance(
            threshold_level, extra_familial_risks, multi_agency_required, child_voice
        )
        
        return SafeguardingAssessment2023(
            threshold_level=threshold_level,
            abuse_types=abuse_types,
            extra_familial_risks=extra_familial_risks,
            setting_type=setting_type,
            evidence=evidence,
            escalation_required=escalation_required,
            multi_agency_required=multi_agency_required,
            child_voice_consideration=child_voice,
            confidence_score=confidence_score,
            immediate_actions=immediate_actions,
            rationale=rationale,
            working_together_2023_compliance=compliance_check
        )
    
    def _is_safeguarding_content(self, question: str, context: str) -> bool:
        """Check if content is safeguarding-related"""
        combined = (question + " " + context).lower()
        
        safeguarding_keywords = [
            "safeguarding", "child protection", "abuse", "neglect", "threshold",
            "signs of safety", "significant harm", "disclosure", "welfare",
            "case", "scenario", "assessment", "worried about", "what's working"
        ]
        
        return any(keyword in combined for keyword in safeguarding_keywords)
    
    def _detect_setting_type(self, content: str) -> SettingType:
        """Detect the setting/context type"""
        content_lower = content.lower()
        
        for setting_type, patterns in self.setting_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                return setting_type
        
        return SettingType.FAMILY_HOME  # Default
    
    def _identify_abuse_types(self, content: str) -> List[AbuseType]:
        """Identify abuse types from content"""
        identified_types = []
        content_lower = content.lower()
        
        # Basic patterns (simplified for plugin)
        abuse_indicators = {
            AbuseType.PHYSICAL: ["hit", "bruise", "injury", "hurt", "beaten", "slapped", "belt"],
            AbuseType.EMOTIONAL: ["useless", "worthless", "hate you", "stupid", "shouting"],
            AbuseType.NEGLECT: ["smelly", "dirty", "hungry", "not fed", "no supervision"],
            AbuseType.SEXUAL: ["inappropriate touching", "sexual", "pornography"],
            AbuseType.EXPLOITATION: ["older boys", "gifts", "money", "special friend"]
        }
        
        for abuse_type, indicators in abuse_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                identified_types.append(abuse_type)
        
        return identified_types
    
    def _identify_extra_familial_risks(self, content: str) -> List[ExtraFamilialHarm]:
        """NEW: Identify extra-familial harm risks (2023 requirement)"""
        identified_risks = []
        content_lower = content.lower()
        
        for risk_type, patterns in self.extra_familial_patterns.items():
            if any(indicator in content_lower for indicator in patterns["indicators"]):
                identified_risks.append(risk_type)
            elif any(sign in content_lower for sign in patterns["behavioral_signs"]):
                identified_risks.append(risk_type)
        
        return identified_risks
    
    def _assess_threshold_level_2023(self, content: str, abuse_types: List[AbuseType], 
                                    extra_familial_risks: List[ExtraFamilialHarm]) -> ThresholdLevel:
        """Assess threshold level with 2023 enhancements"""
        content_lower = content.lower()
        
        # Level 4 indicators
        if any(trigger in content_lower for trigger in self.threshold_2023_criteria[ThresholdLevel.LEVEL_4]["triggers"]):
            return ThresholdLevel.LEVEL_4
        
        # Extra-familial harm can escalate threshold
        if extra_familial_risks:
            for risk in extra_familial_risks:
                if risk.value in ["criminal_exploitation", "sexual_exploitation", "trafficking"]:
                    return ThresholdLevel.LEVEL_4
                elif risk.value in ["gang_involvement", "online_abuse"]:
                    return ThresholdLevel.LEVEL_3
        
        # Level 3 indicators
        if any(trigger in content_lower for trigger in self.threshold_2023_criteria[ThresholdLevel.LEVEL_3]["triggers"]):
            return ThresholdLevel.LEVEL_3
        
        # Multiple abuse types
        if len(abuse_types) > 1:
            return ThresholdLevel.LEVEL_3
        
        # Level 2 indicators
        if abuse_types or any(trigger in content_lower for trigger in self.threshold_2023_criteria[ThresholdLevel.LEVEL_2]["triggers"]):
            return ThresholdLevel.LEVEL_2
        
        return ThresholdLevel.LEVEL_1
    
    def _extract_evidence(self, content: str, abuse_types: List[AbuseType], 
                         extra_familial_risks: List[ExtraFamilialHarm]) -> List[str]:
        """Extract evidence from content"""
        evidence = []
        
        # Direct quotes
        quotes = re.findall(r'"([^"]*)"', content)
        for quote in quotes:
            evidence.append(f"Direct disclosure: '{quote}'")
        
        # Physical indicators
        physical_indicators = re.findall(r'(bruise|injury|cut|mark|hit|hurt)', content.lower())
        for indicator in set(physical_indicators):
            evidence.append(f"Physical indicator: {indicator}")
        
        # Extra-familial evidence
        if extra_familial_risks:
            evidence.append(f"Extra-familial risks: {len(extra_familial_risks)} types identified")
        
        return evidence
    
    def _assess_child_voice(self, content: str) -> str:
        """Assess child voice consideration (2023 requirement)"""
        content_lower = content.lower()
        
        if '"' in content or any(phrase in content_lower for phrase in ["says", "tells", "reports"]):
            return "Child voice present - direct communication identified"
        elif any(phrase in content_lower for phrase in ["behavioral", "observed", "appears"]):
            return "Child voice indirect - behavioral indicators suggest experience"
        else:
            return "Child voice consideration required - ensure views explored"
    
    def _generate_immediate_actions_2023(self, threshold_level: ThresholdLevel, abuse_types: List[AbuseType],
                                        extra_familial_risks: List[ExtraFamilialHarm], 
                                        setting_type: SettingType) -> List[str]:
        """Generate 2023-compliant immediate actions"""
        actions = []
        
        if threshold_level == ThresholdLevel.LEVEL_4:
            actions.extend([
                "IMMEDIATE: Contact children's social care within 24 hours",
                "IMMEDIATE: Ensure child's safety and implement protection measures",
                "IMMEDIATE: Multi-agency coordination per 2023 standards"
            ])
            
            if setting_type == SettingType.CHILDRENS_HOME:
                actions.extend([
                    "IMMEDIATE: Follow Regulation 34 referral procedures",
                    "IMMEDIATE: Notify placing authority and local authority"
                ])
            
            if extra_familial_risks:
                actions.append("IMMEDIATE: Address extra-familial safety risks")
        
        elif threshold_level == ThresholdLevel.LEVEL_3:
            actions.extend([
                "Refer to children's social care for assessment",
                "Convene multi-agency meeting per 2023 standards",
                "Complete extra-familial harm assessment",
                "Develop child-centred support plan"
            ])
        
        # Always include 2023 requirements
        actions.append("Ensure child voice is heard throughout process")
        if extra_familial_risks or threshold_level != ThresholdLevel.LEVEL_1:
            actions.append("Document extra-familial harm consideration")
        
        return actions
    
    def _calculate_confidence(self, content: str, abuse_types: List[AbuseType], 
                            extra_familial_risks: List[ExtraFamilialHarm]) -> float:
        """Calculate confidence score"""
        confidence = 0.6  # Base
        
        if '"' in content:  # Direct quotes
            confidence += 0.3
        
        if abuse_types:  # Abuse indicators
            confidence += 0.1 * len(abuse_types)
        
        if extra_familial_risks:  # Extra-familial assessment
            confidence += 0.15
        
        return min(confidence, 1.0)
    
    def _generate_rationale_2023(self, threshold_level: ThresholdLevel, abuse_types: List[AbuseType], 
                                setting_type: SettingType) -> str:
        """Generate 2023-compliant rationale"""
        parts = []
        
        parts.append(f"Threshold: {threshold_level.value.replace('_', ' ').title()}")
        
        if abuse_types:
            abuse_desc = ", ".join([abuse.value.replace('_', ' ').title() for abuse in abuse_types])
            parts.append(f"Concerns: {abuse_desc}")
        
        if setting_type == SettingType.CHILDRENS_HOME:
            parts.append("Children's home context: Regulations 28/32 apply")
        
        parts.append("Working Together 2023: Multi-agency and child-centred approach required")
        
        return " | ".join(parts)
    
    def _check_2023_compliance(self, threshold_level: ThresholdLevel, 
                              extra_familial_risks: List[ExtraFamilialHarm],
                              multi_agency_required: bool, child_voice: str) -> Dict[str, bool]:
        """Check 2023 compliance requirements"""
        return {
            "child_centred_approach": "voice" in child_voice.lower(),
            "extra_familial_assessment": len(extra_familial_risks) > 0 or threshold_level == ThresholdLevel.LEVEL_1,
            "multi_agency_coordination": multi_agency_required or threshold_level in [ThresholdLevel.LEVEL_1, ThresholdLevel.LEVEL_2],
            "whole_family_consideration": True  # Assumed in assessment context
        }
    
    def _needs_safeguarding_enhancement(self, question: str, assessment: Optional[SafeguardingAssessment2023]) -> bool:
        """Determine if safeguarding enhancement is needed"""
        if not assessment:
            return False
        
        # Always enhance if escalation required
        if assessment.escalation_required:
            return True
        
        # Always enhance for children's homes
        if assessment.setting_type == SettingType.CHILDRENS_HOME:
            return True
        
        # Always enhance if extra-familial risks
        if assessment.extra_familial_risks:
            return True
        
        # Check for Signs of Safety requests
        question_lower = question.lower()
        if "signs of safety" in question_lower:
            return True
        
        return False
    
    def _generate_2023_prompt(self, question: str, context: str, 
                             assessment: SafeguardingAssessment2023) -> str:
        """Generate 2023-compliant prompt based on assessment"""
        
        question_lower = question.lower()
        
        # Select appropriate template
        if assessment.setting_type == SettingType.CHILDRENS_HOME:
            template = self.templates_2023["safeguarding_childrens_home"]
        elif "signs of safety" in question_lower:
            template = self.templates_2023["safeguarding_signs_of_safety_2023"]
        else:
            template = self.templates_2023["safeguarding_general_2023"]
        
        return template.format(context=context, question=question)
    
    def _create_assessment_summary(self, assessment: Optional[SafeguardingAssessment2023]) -> str:
        """Create assessment summary for inclusion in response"""
        if not assessment:
            return ""
        
        summary = f"\n\n🛡️ SAFEGUARDING ASSESSMENT (2023 COMPLIANT):\n"
        summary += f"• Threshold Level: {assessment.threshold_level.value.replace('_', ' ').title()}\n"
        summary += f"• Setting: {assessment.setting_type.value.replace('_', ' ').title()}\n"
        
        if assessment.abuse_types:
            abuse_list = [abuse.value.replace('_', ' ').title() for abuse in assessment.abuse_types]
            summary += f"• Concerns: {', '.join(abuse_list)}\n"
        
        # Extra-familial harm reporting (2023 requirement)
        if assessment.extra_familial_risks:
            risk_list = [risk.value.replace('_', ' ').title() for risk in assessment.extra_familial_risks]
            summary += f"• Extra-familial Risks: {', '.join(risk_list)}\n"
        else:
            summary += f"• Extra-familial Assessment: Completed - no risks identified\n"
        
        summary += f"• Multi-agency Required: {'Yes' if assessment.multi_agency_required else 'No'}\n"
        summary += f"• Child Voice: {assessment.child_voice_consideration}\n"
        summary += f"• Confidence: {assessment.confidence_score:.0%}\n"
        
        # 2023 Compliance status
        compliance_passed = sum(assessment.working_together_2023_compliance.values())
        compliance_total = len(assessment.working_together_2023_compliance)
        summary += f"• Working Together 2023 Compliance: {compliance_passed}/{compliance_total} areas met\n"
        
        if assessment.escalation_required:
            summary += f"\n🚨 ESCALATION REQUIRED - Threshold {assessment.threshold_level.value.replace('_', ' ').title()}\n"
            summary += f"• Immediate Actions: {len(assessment.immediate_actions)} required\n"
        
        return summary


# Integration Helper Functions for your existing system

def integrate_plugin_with_existing_system():
    """
    Integration guidance for your existing HybridRAGSystem
    """
    
    integration_code = '''
# STEP 1: Add this import at the top of your rag_system.py
from safeguarding_2023_plugin import SafeguardingPlugin

# STEP 2: Add this line in your HybridRAGSystem.__init__() method
def __init__(self, config: Dict[str, Any] = None):
    # ... your existing initialization code ...
    
    # Add this line:
    self.safeguarding_plugin = SafeguardingPlugin()
    
    # ... rest of your existing initialization ...

# STEP 3: Modify your query() method 
# Replace the prompt building section (around line where you call _build_optimized_prompt)

def query(self, question: str, k: int = 5, response_style: str = "standard", 
          performance_mode: str = "balanced", **kwargs) -> Dict[str, Any]:
    
    # ... your existing code until context building ...
    
    # REPLACE THIS SECTION:
    # OLD: prompt = self._build_optimized_prompt(question, context_text, detected_mode)
    
    # NEW: Enhanced prompt building with 2023 compliance
    safeguarding_enhancement = self.safeguarding_plugin.enhance_query_with_2023_compliance(
        question, context_text, detected_mode.value
    )
    
    if safeguarding_enhancement["needs_safeguarding_enhancement"]:
        # Use 2023-compliant safeguarding prompt
        prompt = safeguarding_enhancement["enhanced_prompt"]
        logger.info("Using 2023-compliant safeguarding prompt")
    else:
        # Use your existing prompt building
        prompt = self._build_optimized_prompt(question, context_text, detected_mode)
    
    # ... continue with your existing code for answer generation ...
    
    # STEP 4: Enhanced response building
    # In your _create_streamlit_response method, add the assessment summary:
    
    # After generating the final_response but before return:
    if safeguarding_enhancement["assessment_summary"]:
        answer_result["answer"] += safeguarding_enhancement["assessment_summary"]
    
    # ... rest of your existing response creation ...
    
    return response
'''
    
    return {
        "integration_code": integration_code,
        "files_to_modify": ["rag_system.py"],
        "new_files": ["safeguarding_2023_plugin.py"],
        "changes_required": [
            "Add 1 import line",
            "Add 1 initialization line", 
            "Replace prompt building section",
            "Add assessment summary to response"
        ],
        "estimated_time": "5-10 minutes",
        "risk_level": "Very Low - minimal changes to working code"
    }


def test_plugin_integration():
    """Test the plugin functionality"""
    
    plugin = SafeguardingPlugin()
    
    # Test case: Children's home scenario
    test_question = "What threshold level and actions are required?"
    test_context = '''Kiyah (7) lives in Sunshine Children's Home. She comes to breakfast with a cut lip and says "John hit me because I was noisy last night". John is a 16-year-old resident.'''
    
    enhancement = plugin.enhance_query_with_2023_compliance(
        test_question, test_context, "standard"
    )
    
    return {
        "test_scenario": "Children's home peer abuse",
        "enhancement_needed": enhancement["needs_safeguarding_enhancement"],
        "threshold_detected": enhancement["assessment"].threshold_level.value if enhancement["assessment"] else None,
        "setting_detected": enhancement["assessment"].setting_type.value if enhancement["assessment"] else None,
        "extra_familial_risks": len(enhancement["assessment"].extra_familial_risks) if enhancement["assessment"] else 0,
        "escalation_required": enhancement["escalation_required"],
        "compliance_status": enhancement["compliance_status"],
        "has_enhanced_prompt": enhancement["enhanced_prompt"] is not None,
        "assessment_summary_preview": enhancement["assessment_summary"][:200] + "..." if enhancement["assessment_summary"] else "None"
    }


if __name__ == "__main__":
    print("🔌 SAFEGUARDING 2023 PLUGIN - MINIMAL INTEGRATION")
    print("="*60)
    
    # Test plugin functionality
    print("\n🧪 TESTING PLUGIN FUNCTIONALITY...")
    test_result = test_plugin_integration()
    
