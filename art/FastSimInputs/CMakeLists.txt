cet_build_plugin(SavedG4TableDumper art::module
    REG_SOURCE src/SavedG4TableDumper_module.cc
    LIBRARIES REG
      Offline::MCDataProducts
      Offline::DataProducts
      art::Framework_Core
      art::Framework_Principal
      art::Framework_Services_Registry
      art_root_io::tfile_support
      art_root_io::TFileService_service
      art::Persistency_Common
      art::Persistency_Provenance
      art::Utilities
      canvas::canvas
      messagefacility::MF_MessageLogger
      fhiclcpp::fhiclcpp
      fhiclcpp::types
      cetlib::cetlib
      cetlib_except::cetlib_except
      CLHEP::CLHEP
      ROOT::Tree
)
install_source(SUBDIRS src)
install_fhicl(SUBDIRS fcl)
