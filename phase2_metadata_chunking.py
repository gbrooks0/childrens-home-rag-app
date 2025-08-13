# Phase 2: Enhanced Metadata & Intelligent Chunking - FIXED INDENTATION
# Save as: phase2_metadata_chunking.py

import re
import json
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
import hashlib
from datetime import datetime
from pathlib import Path

# =============================================================================
# 1. ENHANCED METADATA EXTRACTION
# =============================================================================

@dataclass
class EnhancedDocumentMetadata:
    """Comprehensive metadata structure for children's home documents"""
    
    # Core identification
    doc_id: str
    title: str
    source_file: str
    document_type: str
    
    # Children's home specific metadata
    provider_name: Optional[str] = None
    inspection_date: Optional[str] = None
    overall_rating: Optional[str] = None
    section_ratings: Dict[str, str] = None
    regulatory_references: List[str] = None
    
    # Content classification
    content_categories: List[str] = None
    target_audience: List[str] = None  # [managers, staff, inspectors, etc.]
    urgency_level: str = "normal"  # critical, high, normal, low
    
    # Technical metadata
    word_count: int = 0
    creation_date: Optional[str] = None
    last_modified: Optional[str] = None
    content_hash: Optional[str] = None
    
    # Relationship mapping
    related_documents: List[str] = None
    supersedes: List[str] = None
    references: List[str] = None
    
    # Quality indicators
    confidence_score: float = 1.0
    completeness_score: float = 1.0
    authority_level: str = "medium"  # official, high, medium, low
    
    def __post_init__(self):
        """Initialize default values for mutable fields"""
        if self.section_ratings is None:
            self.section_ratings = {}
        if self.regulatory_references is None:
            self.regulatory_references = []
        if self.content_categories is None:
            self.content_categories = []
        if self.target_audience is None:
            self.target_audience = []
        if self.related_documents is None:
            self.related_documents = []
        if self.supersedes is None:
            self.supersedes = []
        if self.references is None:
            self.references = []

class EnhancedMetadataExtractor:
    """Extract comprehensive metadata from children's home documents"""
    
    def __init__(self):
        self.document_patterns = {
            'ofsted_report': [
                r'ofsted.*inspection',
                r'children\'s home.*inspection',
                r'inspection.*report',
                r'provider.*overview'
            ],
            'policy_document': [
                r'policy',
                r'procedure',
                r'guidance',
                r'handbook'
            ],
            'regulation': [
                r'children\'s homes.*regulations',
                r'regulation.*\d+',
                r'statutory.*guidance'
            ],
            'training_material': [
                r'training',
                r'course.*material',
                r'learning.*guide'
            ],
            'best_practice': [
                r'best.*practice',
                r'good.*practice',
                r'case.*study'
            ]
        }
        
        self.rating_patterns = {
            'outstanding': r'(?i)outstanding',
            'good': r'(?i)\bgood\b',
            'requires_improvement': r'(?i)requires?\s+improvement',
            'inadequate': r'(?i)inadequate'
        }
        
        self.provider_patterns = [
            r'([A-Z][a-z]+ [A-Z][a-z]+ Homes? (?:Ltd|Limited|LLP))',
            r'([A-Z][a-z]+ [A-Z][a-z]+ Care (?:Ltd|Limited))',
            r'(Creating Lifestyles Homes Limited)',
            r'(Yorkshire Serenity Care Ltd)',
            r'(Restoring Lives Ltd)',
        ]
    
    def extract_metadata(self, content: str, filename: str, 
                        existing_metadata: Dict = None) -> EnhancedDocumentMetadata:
        """Extract comprehensive metadata from document content"""
        
        # Generate document ID
        doc_id = self._generate_doc_id(filename, content)
        
        # Basic extraction
        title = self._extract_title(content, filename)
        document_type = self._classify_document_type(content, filename)
        
        # Children's home specific extraction
        provider_name = self._extract_provider_name(content)
        inspection_date = self._extract_inspection_date(content)
        overall_rating = self._extract_overall_rating(content)
        section_ratings = self._extract_section_ratings(content)
        regulatory_refs = self._extract_regulatory_references(content)
        
        # Content classification
        content_categories = self._classify_content_categories(content, document_type)
        target_audience = self._determine_target_audience(content, document_type)
        urgency_level = self._assess_urgency_level(content, document_type)
        
        # Technical metadata
        word_count = len(content.split())
        content_hash = hashlib.md5(content.encode()).hexdigest()
        creation_date = datetime.now().isoformat()
        
        # Authority level
        authority_level = self._determine_authority_level(filename, content, document_type)
        
        # Quality scores
        confidence_score = self._calculate_confidence_score(content, document_type)
        completeness_score = self._calculate_completeness_score(content, document_type)
        
        return EnhancedDocumentMetadata(
            doc_id=doc_id,
            title=title,
            source_file=filename,
            document_type=document_type,
            provider_name=provider_name,
            inspection_date=inspection_date,
            overall_rating=overall_rating,
            section_ratings=section_ratings,
            regulatory_references=regulatory_refs,
            content_categories=content_categories,
            target_audience=target_audience,
            urgency_level=urgency_level,
            word_count=word_count,
            creation_date=creation_date,
            content_hash=content_hash,
            authority_level=authority_level,
            confidence_score=confidence_score,
            completeness_score=completeness_score
        )
    
    def _generate_doc_id(self, filename: str, content: str) -> str:
        """Generate unique document ID"""
        content_sample = content[:200] if content else ""
        raw_id = f"{filename}_{content_sample}"
        return hashlib.sha256(raw_id.encode()).hexdigest()[:12]
    
    def _extract_title(self, content: str, filename: str) -> str:
        """Extract document title"""
        # Look for title patterns in content
        title_patterns = [
            r'^#\s+(.+)$',
            r'^\s*([A-Z][^.!?]*(?:Policy|Procedure|Guidance|Report))\s*$',
            r'Title:\s*(.+)$',
            r'Subject:\s*(.+)$',
        ]
        
        lines = content.split('\n')[:10]  # Check first 10 lines
        for line in lines:
            for pattern in title_patterns:
                match = re.search(pattern, line, re.MULTILINE | re.IGNORECASE)
                if match:
                    title = match.group(1).strip()
                    if len(title) > 5 and len(title) < 100:
                        return title
        
        # Fallback to filename
        clean_filename = Path(filename).stem.replace('_', ' ').replace('-', ' ')
        return ' '.join(word.capitalize() for word in clean_filename.split())
    
    def _classify_document_type(self, content: str, filename: str) -> str:
        """Classify document type based on content and filename"""
        content_lower = content.lower()
        filename_lower = filename.lower()
        
        for doc_type, patterns in self.document_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower) or re.search(pattern, filename_lower):
                    return doc_type
        
        return 'general_document'
    
    def _extract_provider_name(self, content: str) -> Optional[str]:
        """Extract provider name from content"""
        for pattern in self.provider_patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1).strip()
        return None
    
    def _extract_inspection_date(self, content: str) -> Optional[str]:
        """Extract inspection date"""
        date_patterns = [
            r'inspection.*?(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
            r'inspection.*?(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
            r'conducted.*?(\d{1,2}[/-]\d{1,2}[/-]\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def _extract_overall_rating(self, content: str) -> Optional[str]:
        """Extract overall Ofsted rating"""
        for rating, pattern in self.rating_patterns.items():
            if re.search(f'overall.*{pattern}', content, re.IGNORECASE):
                return rating.replace('_', ' ').title()
        return None
    
    def _extract_section_ratings(self, content: str) -> Dict[str, str]:
        """Extract section-specific ratings"""
        sections = {
            'experiences_progress': r'overall experiences.*?progress',
            'help_protection': r'help.*?protection',
            'leadership_management': r'leadership.*?management'
        }
        
        section_ratings = {}
        for section_key, section_pattern in sections.items():
            section_text = re.search(f'{section_pattern}.*?([a-zA-Z]+)', content, re.IGNORECASE | re.DOTALL)
            if section_text:
                rating_text = section_text.group(1).lower()
                for rating, pattern in self.rating_patterns.items():
                    if re.search(pattern, rating_text):
                        section_ratings[section_key] = rating.replace('_', ' ').title()
                        break
        
        return section_ratings
    
    def _extract_regulatory_references(self, content: str) -> List[str]:
        """Extract regulatory references"""
        patterns = [
            r'Regulation (\d+(?:\.\d+)?)',
            r'Schedule (\d+)',
            r'Section (\d+)',
            r'Children\'s Homes.*Regulations.*(\d{4})'
        ]
        
        references = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            references.extend(matches)
        
        return list(set(references))  # Remove duplicates
    
    def _classify_content_categories(self, content: str, document_type: str) -> List[str]:
        """Classify content into categories"""
        categories = []
        content_lower = content.lower()
        
        # Category patterns
        category_patterns = {
            'safeguarding': ['safeguarding', 'child protection', 'safety', 'risk'],
            'care_planning': ['care plan', 'individual needs', 'assessment'],
            'staff_management': ['staff', 'training', 'supervision', 'recruitment'],
            'education': ['education', 'school', 'learning', 'academic'],
            'health': ['health', 'medical', 'mental health', 'wellbeing'],
            'behavior_management': ['behavior', 'behaviour', 'discipline', 'sanctions'],
            'accommodation': ['accommodation', 'premises', 'environment'],
            'complaints': ['complaint', 'grievance', 'concerns'],
            'finance': ['budget', 'financial', 'funding', 'costs']
        }
        
        for category, keywords in category_patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                categories.append(category)
        
        # Document type as category
        if document_type != 'general_document':
            categories.append(document_type)
        
        return categories
    
    def _determine_target_audience(self, content: str, document_type: str) -> List[str]:
        """Determine target audience"""
        audiences = []
        content_lower = content.lower()
        
        audience_patterns = {
            'managers': ['manager', 'management', 'leadership', 'head'],
            'staff': ['staff', 'worker', 'employee', 'team'],
            'inspectors': ['inspector', 'ofsted', 'regulator'],
            'children': ['child', 'young person', 'resident'],
            'parents': ['parent', 'family', 'carer'],
            'professionals': ['social worker', 'professional', 'agency']
        }
        
        for audience, keywords in audience_patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                audiences.append(audience)
        
        # Default audiences based on document type
        defaults = {
            'ofsted_report': ['managers', 'inspectors'],
            'policy_document': ['managers', 'staff'],
            'regulation': ['managers', 'professionals'],
            'training_material': ['staff']
        }
        
        if document_type in defaults:
            for default_audience in defaults[document_type]:
                if default_audience not in audiences:
                    audiences.append(default_audience)
        
        return audiences or ['general']
    
    def _assess_urgency_level(self, content: str, document_type: str) -> str:
        """Assess urgency level of document"""
        content_lower = content.lower()
        
        critical_indicators = ['immediate', 'urgent', 'critical', 'emergency', 'inadequate']
        high_indicators = ['requires improvement', 'action needed', 'must', 'enforcement']
        low_indicators = ['guidance', 'information', 'reference']
        
        if any(indicator in content_lower for indicator in critical_indicators):
            return 'critical'
        elif any(indicator in content_lower for indicator in high_indicators):
            return 'high'
        elif any(indicator in content_lower for indicator in low_indicators):
            return 'low'
        
        return 'normal'
    
    def _determine_authority_level(self, filename: str, content: str, document_type: str) -> str:
        """Determine authority level of source"""
        filename_lower = filename.lower()
        content_lower = content.lower()
        
        # Official sources
        if any(source in filename_lower for source in ['ofsted', 'gov.uk', 'dfe']):
            return 'official'
        
        # High authority indicators
        if any(indicator in content_lower for indicator in ['regulation', 'statutory', 'legal requirement']):
            return 'high'
        
        # Document type authority
        authority_map = {
            'ofsted_report': 'official',
            'regulation': 'official',
            'policy_document': 'high',
            'best_practice': 'medium'
        }
        
        return authority_map.get(document_type, 'medium')
    
    def _calculate_confidence_score(self, content: str, document_type: str) -> float:
        """Calculate confidence score for metadata extraction"""
        score = 0.5  # Base score
        
        # Content length factor
        word_count = len(content.split())
        if word_count > 1000:
            score += 0.2
        elif word_count > 500:
            score += 0.1
        
        # Structure indicators
        if re.search(r'^\s*#', content, re.MULTILINE):  # Has headings
            score += 0.1
        
        if document_type != 'general_document':
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_completeness_score(self, content: str, document_type: str) -> float:
        """Calculate completeness score"""
        score = 0.5
        
        # Expected elements for Ofsted reports
        if document_type == 'ofsted_report':
            expected_elements = ['provider', 'inspection', 'rating', 'recommendation']
            found_elements = sum(1 for element in expected_elements if element in content.lower())
            score = found_elements / len(expected_elements)
        
        # For policies
        elif document_type == 'policy_document':
            expected_elements = ['purpose', 'procedure', 'responsibility', 'review']
            found_elements = sum(1 for element in expected_elements if element in content.lower())
            score = found_elements / len(expected_elements)
        
        return min(score, 1.0)

# =============================================================================
# 2. SIMPLIFIED INTEGRATION CLASS
# =============================================================================

class Phase2EnhancedProcessor:
    """Phase 2 processor for enhanced metadata and chunking"""
    
    def __init__(self):
        self.metadata_extractor = EnhancedMetadataExtractor()
        self.metadata_cache = {}
    
    def enhance_retrieval_with_metadata(self, query: str, retrieved_docs: List[Dict]) -> List[Dict]:
        """Enhance retrieval results using metadata"""
        
        enhanced_docs = []
        
        for doc in retrieved_docs:
            # Get or extract metadata
            doc_metadata = self._get_or_extract_metadata(doc)
            
            # Calculate relevance boost based on metadata
            relevance_boost = self._calculate_metadata_relevance_boost(query, doc_metadata)
            
            # Enhance document with metadata
            enhanced_doc = dict(doc)
            enhanced_doc['metadata'] = asdict(doc_metadata) if doc_metadata else {}
            enhanced_doc['relevance_boost'] = relevance_boost
            enhanced_doc['enhanced_score'] = doc.get('score', 0.5) + relevance_boost
            
            enhanced_docs.append(enhanced_doc)
        
        # Sort by enhanced score
        enhanced_docs.sort(key=lambda x: x.get('enhanced_score', 0), reverse=True)
        
        return enhanced_docs
    
    def _get_or_extract_metadata(self, doc: Dict) -> Optional[EnhancedDocumentMetadata]:
        """Get metadata from cache or extract on demand"""
        
        # Try to get from existing metadata
        existing_metadata = doc.get('metadata', {})
        if existing_metadata and existing_metadata.get('doc_id'):
            doc_id = existing_metadata['doc_id']
            if doc_id in self.metadata_cache:
                return self.metadata_cache[doc_id]
        
        # Extract on demand
        content = doc.get('page_content', doc.get('content', ''))
        filename = doc.get('source', 'unknown_document')
        
        if content:
            metadata = self.metadata_extractor.extract_metadata(content, filename)
            self.metadata_cache[metadata.doc_id] = metadata
            return metadata
        
        return None
    
    def _calculate_metadata_relevance_boost(self, query: str, metadata: EnhancedDocumentMetadata) -> float:
        """Calculate relevance boost based on metadata alignment with query"""
        
        if not metadata:
            return 0.0
        
        boost = 0.0
        query_lower = query.lower()
        
        # Document type relevance
        type_boosts = {
            'ofsted_report': 0.3 if any(word in query_lower for word in ['ofsted', 'inspection', 'rating']) else 0.0,
            'policy_document': 0.3 if any(word in query_lower for word in ['policy', 'procedure', 'guidance']) else 0.0,
            'regulation': 0.4 if any(word in query_lower for word in ['regulation', 'requirement', 'legal']) else 0.0,
            'training_material': 0.2 if 'training' in query_lower else 0.0
        }
        boost += type_boosts.get(metadata.document_type, 0.0)
        
        # Provider name match
        if metadata.provider_name and metadata.provider_name.lower() in query_lower:
            boost += 0.4
        
        # Content category relevance
        for category in metadata.content_categories:
            if category.replace('_', ' ') in query_lower:
                boost += 0.2
        
        # Authority level boost
        authority_boosts = {'official': 0.3, 'high': 0.2, 'medium': 0.1, 'low': 0.0}
        boost += authority_boosts.get(metadata.authority_level, 0.0)
        
        # Urgency level consideration
        urgency_boosts = {'critical': 0.2, 'high': 0.1, 'normal': 0.0, 'low': -0.1}
        boost += urgency_boosts.get(metadata.urgency_level, 0.0)
        
        return min(boost, 0.5)  # Cap boost at 0.5

# =============================================================================
# 3. INTEGRATION FUNCTION
# =============================================================================

def integrate_phase2_with_existing_rag(rag_system):
    """Integrate Phase 2 enhancements with your existing RAG system"""
    
    # Check if Phase 2 already integrated
    if hasattr(rag_system, '_phase2_enhanced'):
        print("⚡ Phase 2 already integrated - skipping duplicate integration")
        return rag_system
    
    print("🚀 INTEGRATING Phase 2 enhancements (Metadata & Chunking)...")
    
    # Add Phase 2 processor
    if not hasattr(rag_system, 'phase2_processor'):
        rag_system.phase2_processor = Phase2EnhancedProcessor()
        print("✅ Phase 2 processor added")
    
    # Store original query method if not already stored
    if not hasattr(rag_system, '_original_query_phase1'):
        if hasattr(rag_system, '_original_query'):
            rag_system._original_query_phase1 = rag_system._original_query
        else:
            rag_system._original_query_phase1 = rag_system.query
        print("✅ Original query method backed up for Phase 2")
    
    # Get current query method (could be Phase 1 enhanced)
    current_query_method = rag_system.query
    
    # Create Phase 2 enhanced query method
    def phase2_enhanced_query(question, k=5, response_style="standard", performance_mode="balanced",
                             is_file_analysis=False, uploaded_files=None, uploaded_images=None):
        """Phase 2 enhanced query with metadata-aware retrieval"""
        
        # Get base results (from Phase 1 if active, or original)
        original_result = current_query_method(
            question=question,
            k=k*2,  # Get more results for better filtering
            response_style=response_style,
            performance_mode=performance_mode,
            is_file_analysis=is_file_analysis,
            uploaded_files=uploaded_files,
            uploaded_images=uploaded_images
        )
        
        # Apply Phase 2 enhancements
        try:
            sources = original_result.get('sources', [])
            if sources:
                print(f"🔍 APPLYING Phase 2 metadata enhancement to {len(sources)} sources")
                
                # Enhance retrieval with metadata
                enhanced_sources = rag_system.phase2_processor.enhance_retrieval_with_metadata(
                    question, sources
                )
                
                # Take top k results after enhancement
                original_result['sources'] = enhanced_sources[:k]
                
                # Add Phase 2 metadata
                original_result['phase2_data'] = {
                    'metadata_enhanced': True,
                    'original_source_count': len(sources),
                    'enhanced_source_count': len(enhanced_sources),
                    'top_source_types': [
                        src.get('metadata', {}).get('document_type', 'unknown') 
                        for src in enhanced_sources[:3]
                    ],
                    'relevance_boosts': [
                        src.get('relevance_boost', 0.0) 
                        for src in enhanced_sources[:3]
                    ]
                }
                
                print(f"✅ ENHANCED {len(enhanced_sources)} sources with metadata")
            else:
                print("⚡ SKIPPING Phase 2 enhancement (no sources)")
                
        except Exception as e:
            print(f"⚠️ Phase 2 enhancement failed, using original: {e}")
        
        return original_result
    
    # Replace query method
    rag_system.query = phase2_enhanced_query
    rag_system._phase2_enhanced = True
    
    print("✅ PHASE 2 integration complete")
    return rag_system

# =============================================================================
# 4. TESTING FUNCTIONS
# =============================================================================

def test_phase2_integration(rag_system):
    """Test Phase 2 integration"""
    
    test_results = {
        'phase2_active': hasattr(rag_system, '_phase2_enhanced'),
        'processor_available': hasattr(rag_system, 'phase2_processor'),
        'backup_preserved': hasattr(rag_system, '_original_query_phase1'),
        'metadata_extractor_working': False,
        'both_phases_active': False
    }
    
    try:
        if hasattr(rag_system, 'phase2_processor'):
            processor = rag_system.phase2_processor
            
            # Test metadata extraction
            test_content = "This is a test Ofsted inspection report for Creating Lifestyles Homes Limited. Overall rating: Good."
            metadata = processor.metadata_extractor.extract_metadata(test_content, "test_report.pdf")
            test_results['metadata_extractor_working'] = metadata.document_type == 'ofsted_report'
            test_results['test_metadata'] = {
                'document_type': metadata.document_type,
                'provider_name': metadata.provider_name,
                'overall_rating': metadata.overall_rating
            }
        
        # Check if both phases are active
        test_results['both_phases_active'] = (
            hasattr(rag_system, '_phase1_enhanced') and 
            hasattr(rag_system, '_phase2_enhanced')
        )
        
    except Exception as e:
        test_results['error'] = str(e)
    
    return test_results
