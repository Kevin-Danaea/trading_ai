"""
FileCleanupService - Servicio de Limpieza de Archivos Temporales y Reportes
============================================================================

Permite eliminar archivos temporales, reportes de validación y archivos de prueba
según antigüedad, patrón de nombre o tipo. Útil para mantener el sistema limpio.
"""

import os
import glob
import logging
from datetime import datetime, timedelta
from typing import Optional, List

logger = logging.getLogger("file_cleanup")

class FileCleanupService:
    """
    Servicio para limpiar archivos temporales y reportes antiguos/no usados.
    """
    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir

    def cleanup_files(self, 
                      pattern: str, 
                      older_than_days: Optional[int] = None,
                      dry_run: bool = False) -> List[str]:
        """
        Elimina archivos que coincidan con el patrón y opcionalmente sean antiguos.
        Args:
            pattern: Patrón glob relativo al base_dir (ej: 'logs/validation_reports/*.json')
            older_than_days: Si se especifica, solo elimina archivos más viejos que estos días
            dry_run: Si True, solo lista los archivos que se eliminarían
        Returns:
            Lista de archivos eliminados
        """
        deleted = []
        now = datetime.now()
        full_pattern = os.path.join(self.base_dir, pattern)
        for file_path in glob.glob(full_pattern):
            try:
                if older_than_days is not None:
                    mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if (now - mtime).days < older_than_days:
                        continue
                if not dry_run:
                    os.remove(file_path)
                    logger.info(f"🗑️ Archivo eliminado: {file_path}")
                deleted.append(file_path)
            except Exception as e:
                logger.warning(f"No se pudo eliminar {file_path}: {e}")
        return deleted

    def cleanup_validation_reports(self, days: int = 30, dry_run: bool = False) -> List[str]:
        """
        Elimina reportes de validación de más de X días.
        """
        return self.cleanup_files('logs/validation_reports/*.json', older_than_days=days, dry_run=dry_run)

    def cleanup_temp_files(self, dry_run: bool = False) -> List[str]:
        """
        Elimina archivos temporales comunes (*.tmp, *.temp, *.bak).
        """
        patterns = ['*.tmp', '*.temp', '*.bak']
        deleted = []
        for pat in patterns:
            deleted += self.cleanup_files(f'**/{pat}', dry_run=dry_run)
        return deleted

    def cleanup_test_reports(self, dry_run: bool = False) -> List[str]:
        """
        Elimina todos los reportes de validación generados por pruebas.
        """
        return self.cleanup_files('logs/validation_reports/validation_report_Test*.json', dry_run=dry_run) 